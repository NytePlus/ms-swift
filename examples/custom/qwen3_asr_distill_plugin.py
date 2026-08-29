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
from typing import Any, Callable, Dict, Optional

import torch
from torch import nn
from transformers.masking_utils import AttentionMaskInterface, eager_mask
from transformers.modeling_utils import AttentionInterface

from swift.model import MODEL_MAPPING, MLLMModelType, ModelArch, register_model
from swift.model.models.qwen import Qwen3ASRLoader

BACKEND_NAME = 'applicable_attention'
ATTENTION_TRANSFORMS: Dict[str, Callable] = {}


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
                attention._applicable_attention_capture = selected and capture and scope == 'thinker_text'
                attention._applicable_attention_config = config
                attention._applicable_attention_weights = None


qwen3_asr_meta = deepcopy(MODEL_MAPPING[MLLMModelType.qwen3_asr])
qwen3_asr_meta.loader = Qwen3ASRAttentionLoader
qwen3_asr_meta.model_arch = ModelArch.qwen3_asr
register_model(qwen3_asr_meta, exist_ok=True)
