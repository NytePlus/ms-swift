# Copyright (c) ModelScope Contributors. All rights reserved.
"""External Qwen3-ASR attention backend and loader for ms-swift.

Load with ``--external_plugins examples/custom/qwen3_asr_distill_plugin.py``.
Custom configuration lives under ``--model_kwargs`` keys
``applicable_attention`` and ``softdtw_distill``.
"""

import ast
import json
import os
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Set

import torch
from torch import nn
from transformers.masking_utils import AttentionMaskInterface, eager_mask
from transformers.modeling_utils import AttentionInterface

BACKEND_NAME = 'applicable_attention'
ATTENTION_TRANSFORMS: Dict[str, Callable] = {}
_ATTENTION_INTERVENTION_RUNTIME = None


class AttentionInterventionRuntime:
    """Mutable per-process state for test-time T<-C attention interventions."""

    TEACHER_STRATEGIES = {'correct_teacher', 'shuffled_teacher'}
    STRATEGIES = {'baseline', 'correct_teacher', 'shuffled_teacher', 'zero_context', 'matched_mass_random'}

    def __init__(self, strategy: str, *, alpha: float = 1.0, layers=None, seed: int = 42):
        if strategy not in self.STRATEGIES:
            raise ValueError(f'Unknown intervention strategy {strategy!r}; available={sorted(self.STRATEGIES)}')
        self.strategy = strategy
        self.alpha = float(alpha)
        self.layers: Optional[Set[int]] = _layer_set(layers)
        self.seed = int(seed)
        self.mode = 'student'
        self.step = 0
        self.student_context_mask: Optional[torch.Tensor] = None
        self.teacher_context_mask: Optional[torch.Tensor] = None
        self.teacher_weights: Dict[int, torch.Tensor] = {}
        self.stats = {
            'teacher_captures': 0,
            'applied_rows': 0,
            'max_abs_delta': 0.0,
            'context_mass_before': 0.0,
            'context_mass_after': 0.0,
        }

    @property
    def needs_teacher(self) -> bool:
        return self.strategy in self.TEACHER_STRATEGIES

    def selected(self, layer_index: Optional[int]) -> bool:
        return layer_index is not None and (self.layers is None or layer_index in self.layers)

    def begin_teacher(self, context_mask: torch.Tensor):
        self.mode = 'teacher'
        self.teacher_context_mask = context_mask.bool()
        self.teacher_weights.clear()

    def begin_student(self, context_mask: torch.Tensor):
        self.mode = 'student'
        self.student_context_mask = context_mask.bool()

    def advance(self):
        self.step += 1


def set_attention_intervention_runtime(runtime: Optional[AttentionInterventionRuntime]):
    global _ATTENTION_INTERVENTION_RUNTIME
    _ATTENTION_INTERVENTION_RUNTIME = runtime


def get_attention_intervention_runtime() -> Optional[AttentionInterventionRuntime]:
    return _ATTENTION_INTERVENTION_RUNTIME


def clear_attention_intervention_runtime():
    set_attention_intervention_runtime(None)


def register_attention_transform(name: str, transform: Callable, *, exist_ok: bool = False):
    """Register a pre-softmax logits transform used by applicable_attention_forward."""
    if not exist_ok and name in ATTENTION_TRANSFORMS:
        raise ValueError(f'Attention transform {name!r} is already registered.')
    ATTENTION_TRANSFORMS[name] = transform


def identity_attention_transform(logits: torch.Tensor, **kwargs) -> torch.Tensor:
    return logits


def scale_attention_transform(logits: torch.Tensor, *, config: Dict[str, Any], **kwargs) -> torch.Tensor:
    return logits * float(config.get('scale', 1.0))


register_attention_transform('identity', identity_attention_transform)
register_attention_transform('scale', scale_attention_transform)


def _repeat_kv(states: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return states
    batch, heads, length, width = states.shape
    states = states[:, :, None, :, :].expand(batch, heads, repeats, length, width)
    return states.reshape(batch, heads * repeats, length, width)


def _context_indices(mask: Optional[torch.Tensor], batch_index: int, key_length: int, device):
    if mask is None or batch_index >= mask.shape[0]:
        return torch.empty(0, dtype=torch.long, device=device)
    row = mask[batch_index].to(device=device)
    if row.shape[0] < key_length:
        row = nn.functional.pad(row, (0, key_length - row.shape[0]), value=False)
    return row[:key_length].nonzero(as_tuple=False).flatten()


def _teacher_target(weights, teacher_weights, student_indices, teacher_indices, alpha):
    eps = torch.finfo(torch.float32).eps
    source = weights.float()
    teacher = teacher_weights.float()
    student_context = source.index_select(-1, student_indices)
    teacher_context = teacher.index_select(-1, teacher_indices)
    if teacher_context.shape[-1] != student_context.shape[-1]:
        teacher_context = nn.functional.interpolate(
            teacher_context.unsqueeze(0), size=student_context.shape[-1], mode='linear', align_corners=False).squeeze(0)

    teacher_mass = teacher_context.sum(dim=-1, keepdim=True).clamp(min=eps, max=1 - eps)
    teacher_conditional = teacher_context / teacher_context.sum(dim=-1, keepdim=True).clamp_min(eps)
    target = source.clone()
    target_context = teacher_conditional * teacher_mass
    target.index_copy_(-1, student_indices, target_context)

    non_context_mask = torch.ones(source.shape[-1], dtype=torch.bool, device=source.device)
    non_context_mask[student_indices] = False
    non_context_indices = non_context_mask.nonzero(as_tuple=False).flatten()
    if non_context_indices.numel():
        non_context = source.index_select(-1, non_context_indices)
        non_context = non_context / non_context.sum(dim=-1, keepdim=True).clamp_min(eps)
        target.index_copy_(-1, non_context_indices, non_context * (1 - teacher_mass))

    # Geometric interpolation is identity at alpha=0, exact replacement at alpha=1,
    # and remains a valid distribution for dose-response values above one.
    blended = torch.exp((1 - alpha) * source.clamp_min(eps).log() + alpha * target.clamp_min(eps).log())
    return (blended / blended.sum(dim=-1, keepdim=True).clamp_min(eps)).to(weights.dtype)


def _apply_attention_intervention(weights: torch.Tensor, runtime: AttentionInterventionRuntime, layer_index: int):
    if runtime.strategy == 'baseline' or not runtime.selected(layer_index):
        return weights
    result = weights.clone()
    query_index = result.shape[-2] - 1
    for batch_index in range(result.shape[0]):
        student_indices = _context_indices(
            runtime.student_context_mask, batch_index, result.shape[-1], result.device)
        if not student_indices.numel():
            continue
        row = result[batch_index, :, query_index, :]
        original_row = row.clone()
        original_context_mass = original_row.index_select(-1, student_indices).sum().item()
        if runtime.strategy == 'zero_context':
            row.index_fill_(-1, student_indices, 0)
            row.div_(row.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(row.dtype).tiny))
        elif runtime.strategy == 'matched_mass_random':
            generator = torch.Generator(device=result.device)
            generator.manual_seed(runtime.seed + 1009 * runtime.step + 9176 * layer_index + batch_index)
            permutation = torch.randperm(student_indices.numel(), generator=generator, device=result.device)
            context = row.index_select(-1, student_indices)
            row.index_copy_(-1, student_indices, context.index_select(-1, permutation))
        else:
            try:
                teacher = runtime.teacher_weights[layer_index][batch_index, :, 0, :]
            except (KeyError, IndexError) as error:
                raise RuntimeError(f'Missing teacher attention for layer {layer_index} at step {runtime.step}.') from error
            teacher_indices = _context_indices(
                runtime.teacher_context_mask, batch_index, teacher.shape[-1], teacher.device)
            if not teacher_indices.numel():
                continue
            row.copy_(_teacher_target(row, teacher, student_indices, teacher_indices, runtime.alpha))
        delta = (row - original_row).abs().max().item()
        if delta > 0:
            runtime.stats['applied_rows'] += 1
            runtime.stats['max_abs_delta'] = max(runtime.stats['max_abs_delta'], delta)
            runtime.stats['context_mass_before'] += original_context_mass
            runtime.stats['context_mass_after'] += row.index_select(-1, student_indices).sum().item()
    return result


def applicable_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Eager attention backend with a registered transform immediately before softmax."""
    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)
    logits = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    config = getattr(module, '_applicable_attention_config', {})
    if getattr(module, '_applicable_attention_enabled', False):
        transform_name = str(config.get('transform', 'identity'))
        try:
            transform = ATTENTION_TRANSFORMS[transform_name]
        except KeyError as error:
            raise ValueError(
                f'Unknown attention transform {transform_name!r}; available={list(ATTENTION_TRANSFORMS)}') from error
        logits = transform(logits, module=module, attention_mask=attention_mask, config=config, **kwargs)

    if attention_mask is not None:
        logits = logits + attention_mask[:, :, :, :key_states.shape[-2]]
    weights = nn.functional.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
    runtime = get_attention_intervention_runtime()
    layer_index = getattr(module, '_applicable_attention_layer_index', getattr(module, 'layer_idx', None))
    if runtime is not None and runtime.selected(layer_index):
        if runtime.mode == 'teacher':
            runtime.teacher_weights[layer_index] = weights[:, :, -1:, :].detach()
            runtime.stats['teacher_captures'] += weights.shape[0]
        else:
            weights = _apply_attention_intervention(weights, runtime, layer_index)
    weights = nn.functional.dropout(weights, p=dropout, training=module.training)
    output = torch.matmul(weights, value_states).transpose(1, 2).contiguous()

    if getattr(module, '_applicable_attention_capture', False):
        # Deliberately retain the student graph. SoftDTWDistillLoss clears this after each loss call.
        module._applicable_attention_weights = weights
    return output, weights


AttentionInterface.register(BACKEND_NAME, applicable_attention_forward)
AttentionMaskInterface.register(BACKEND_NAME, eager_mask)


def _parse_config(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return deepcopy(value)
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        parsed = ast.literal_eval(value)
    if not isinstance(parsed, dict):
        raise TypeError(f'Expected a JSON object, got {type(parsed).__name__}.')
    return parsed


def _custom_config(name: str, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    direct = model_kwargs.pop(name, None)
    return _parse_config(direct if direct is not None else os.environ.get(name.upper()))


def _layer_set(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = [part.strip() for part in value.split(',') if part.strip()]
    if isinstance(value, int):
        value = [value]
    return {int(index) for index in value}


def _scope_set(value):
    if value is None:
        return {'thinker_text'}
    if isinstance(value, str):
        value = [part.strip() for part in value.split(',') if part.strip()]
    scopes = set(value)
    if 'all' in scopes:
        return {'thinker_text', 'audio_encoder'}
    unknown = scopes - {'thinker_text', 'audio_encoder'}
    if unknown:
        raise ValueError(f'Unknown applicable_attention scope: {sorted(unknown)}')
    return scopes


if os.environ.get('QWEN3_ASR_ATTENTION_STANDALONE') != '1':
    from swift.model import MODEL_MAPPING, MLLMModelType, ModelArch, register_model
    from swift.model.models.qwen import Qwen3ASRLoader

    class Qwen3ASRAttentionLoader(Qwen3ASRLoader):
        """Official Qwen3-ASR loader wrapper that consumes custom attention JSON before model loading."""

        def __init__(self, *args, model_kwargs=None, **kwargs):
            clean_model_kwargs = dict(model_kwargs or {})
            self.attention_config = _custom_config('applicable_attention', clean_model_kwargs)
            self.distill_config = _custom_config('softdtw_distill', clean_model_kwargs)
            enabled = bool(self.attention_config.get('enabled', bool(self.distill_config)))
            self.attention_config['enabled'] = enabled
            if enabled:
                kwargs['attn_impl'] = BACKEND_NAME
            super().__init__(*args, model_kwargs=clean_model_kwargs, **kwargs)

        def get_model(self, model_dir: str, config, processor, model_kwargs):
            # Defensive removal for direct get_model_processor(..., model_kwargs={...}) callers.
            model_kwargs = dict(model_kwargs)
            model_kwargs.pop('applicable_attention', None)
            model_kwargs.pop('softdtw_distill', None)
            model = super().get_model(model_dir, config, processor, model_kwargs)
            if self.attention_config.get('enabled'):
                self._configure_modules(model)
            return model

        def _configure_modules(self, model):
            config = deepcopy(self.attention_config)
            scopes = _scope_set(config.get('scope'))
            configured_layers = config.get('layers', self.distill_config.get('layers'))
            layers = _layer_set(configured_layers)
            capture = bool(config.get('capture', bool(self.distill_config)))

            groups = {}
            if 'thinker_text' in scopes:
                groups['thinker_text'] = model.thinker.model.layers
            if 'audio_encoder' in scopes:
                groups['audio_encoder'] = model.thinker.audio_tower.layers

            for scope, module_layers in groups.items():
                for index, layer in enumerate(module_layers):
                    selected = layers is None or index in layers
                    attention = layer.self_attn
                    attention._applicable_attention_enabled = selected
                    attention._applicable_attention_layer_index = index
                    attention._applicable_attention_capture = selected and capture and scope == 'thinker_text'
                    attention._applicable_attention_config = config
                    attention._applicable_attention_weights = None

    qwen3_asr_meta = deepcopy(MODEL_MAPPING[MLLMModelType.qwen3_asr])
    qwen3_asr_meta.loader = Qwen3ASRAttentionLoader
    qwen3_asr_meta.model_arch = ModelArch.qwen3_asr
    register_model(qwen3_asr_meta, exist_ok=True)
