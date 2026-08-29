# Copyright (c) ModelScope Contributors. All rights reserved.
"""Soft-DTW attention-map distillation for Qwen3-ASR.

The student remains owned by the official ms-swift trainer.  The teacher is
loaded once by :class:`TeacherRunner`, is permanently frozen, and is never
attached to the student model or its optimizer.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .base import BaseLoss

logger = logging.getLogger(__name__)


def _softmin(values: torch.Tensor, gamma: float) -> torch.Tensor:
    """Differentiable minimum that remains stable when values contain infinity."""
    return -gamma * torch.logsumexp(-values / gamma, dim=0)


def soft_dtw(cost: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Return the differentiable Soft-DTW value for a 2-D pairwise cost matrix."""
    if cost.ndim != 2 or min(cost.shape) == 0:
        raise ValueError(f'cost must be a non-empty 2-D tensor, got shape={tuple(cost.shape)}')
    if gamma <= 0:
        raise ValueError(f'gamma must be positive, got {gamma}')

    rows, cols = cost.shape
    table = cost.new_full((rows + 1, cols + 1), float('inf'))
    table[0, 0] = 0
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            predecessors = torch.stack((table[i - 1, j], table[i, j - 1], table[i - 1, j - 1]))
            table[i, j] = cost[i - 1, j - 1] + _softmin(predecessors, gamma)
    return table[rows, cols]


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_layers(value: Any, default: Sequence[int] = (0, 1, 2)) -> List[int]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        value = [item.strip() for item in value.split(',') if item.strip()]
    if isinstance(value, int):
        value = [value]
    return [int(item) for item in value]


def _get_tokenizer(processor):
    return getattr(processor, 'tokenizer', processor)


def _strip_asr_prefix(text: str) -> str:
    marker = '<asr_text>'
    marker_index = text.find(marker)
    return text[marker_index + len(marker):].strip() if marker_index >= 0 else text.strip()


def _extract_context_transcript(input_ids: torch.Tensor, tokenizer) -> Tuple[Optional[str], Optional[str]]:
    text = tokenizer.decode(input_ids.tolist(), skip_special_tokens=False)
    context = None
    system_match = re.search(r'<\|im_start\|>system\s*\n(.*?)(?:<\|im_end\|>|$)', text, re.DOTALL)
    if system_match:
        context = system_match.group(1).strip()
    if context is None:
        legacy_match = re.search(r'Context is:\s*(.*?)\n', text, re.DOTALL)
        if legacy_match:
            context = legacy_match.group(1).strip()
            if context.lower() == 'none':
                context = ''

    transcript = None
    assistant_match = re.search(r'<\|im_start\|>assistant\s*\n(.*?)(?:<\|im_end\|>|$)', text, re.DOTALL)
    if assistant_match:
        assistant_text = re.sub(r'<think>.*?</think>\s*\n?', '', assistant_match.group(1), flags=re.DOTALL)
        transcript = _strip_asr_prefix(assistant_text)
    return context, transcript


def _find_subsequence(sequence: Sequence[int], subsequence: Sequence[int]) -> Optional[List[int]]:
    width = len(subsequence)
    if width == 0:
        return None
    for start in range(len(sequence) - width + 1):
        if list(sequence[start:start + width]) == list(subsequence):
            return list(range(start, start + width))
    return None


def _build_teacher_ids(context: str, transcript: str, tokenizer):
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    transcript_ids = tokenizer.encode(transcript, add_special_tokens=False)
    separator_ids = tokenizer.encode('\n', add_special_tokens=False)
    if not context_ids or not transcript_ids:
        return None
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    ids = bos + context_ids + separator_ids + transcript_ids
    context_start = len(bos)
    transcript_start = context_start + len(context_ids) + len(separator_ids)
    return {
        'ids': ids,
        'context_positions': list(range(context_start, context_start + len(context_ids))),
        'transcript_positions': list(range(transcript_start, transcript_start + len(transcript_ids))),
    }


def _pad_batch(rows: Sequence[Sequence[int]], pad_token_id: int, device: torch.device):
    max_length = max(map(len, rows))
    padded, masks = [], []
    for row in rows:
        padding = max_length - len(row)
        padded.append(list(row) + [pad_token_id] * padding)
        masks.append([1] * len(row) + [0] * padding)
    return (
        torch.tensor(padded, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.long, device=device),
    )


@dataclass
class TeacherOutput:
    attentions: Sequence[Optional[torch.Tensor]]


class TeacherRunner:
    """Load and run one frozen text teacher without creating a second trainer."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_id = config.get('teacher_model')
        if not self.model_id:
            raise ValueError('model_kwargs.softdtw_distill.teacher_model is required.')
        self.model = None
        self.tokenizer = None
        self.device = None
        self.pad_token_id = None

    @staticmethod
    def _dtype(name: str):
        aliases = {
            'auto': 'auto',
            'float16': torch.float16,
            'fp16': torch.float16,
            'bfloat16': torch.bfloat16,
            'bf16': torch.bfloat16,
            'float32': torch.float32,
            'fp32': torch.float32,
        }
        if name not in aliases:
            raise ValueError(f'Unsupported teacher_dtype: {name}')
        return aliases[name]

    def _load(self, device: torch.device):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        trust_remote_code = bool(self.config.get('trust_remote_code', True))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=self._dtype(str(self.config.get('teacher_dtype', 'bfloat16')).lower()),
            attn_implementation='eager',
            trust_remote_code=trust_remote_code,
        )
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.eos_token_id
        logger.info('Loaded frozen Soft-DTW teacher from %s on %s.', self.model_id, device)

    def ensure_loaded(self, device: torch.device):
        if self.model is None:
            self._load(device)
        elif self.device != device:
            self.model.to(device)
            self.device = device

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> TeacherOutput:
        self.ensure_loaded(input_ids.device)
        self.model.eval()
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True, use_cache=False)
        return TeacherOutput(attentions=outputs.attentions)


class SoftDTWDistillLoss(BaseLoss):
    """CE plus Soft-DTW alignment of student audio-to-context and teacher text attention."""

    def __init__(self, args, trainer):
        super().__init__(args, trainer)
        config = _as_dict(_as_dict(getattr(args, 'model_kwargs', {})).get('softdtw_distill'))
        self.alpha = float(config.get('alpha', 0.1))
        self.gamma = float(config.get('gamma', 1.0))
        self.cost_metric = str(config.get('cost', 'kl')).lower()
        if self.cost_metric not in {'kl', 'l2'}:
            raise ValueError(f'cost must be "kl" or "l2", got {self.cost_metric}')
        self.layers = _as_layers(config.get('layers'))
        self.teacher_layers = _as_layers(config.get('teacher_layers'), self.layers)
        self.teacher = TeacherRunner(config)
        self.trainer_ref = trainer

        if getattr(args, 'gradient_checkpointing', False):
            checkpoint_kwargs = dict(getattr(args, 'gradient_checkpointing_kwargs', None) or {})
            if checkpoint_kwargs.get('use_reentrant') is True:
                raise ValueError('softdtw_distill requires gradient_checkpointing_kwargs.use_reentrant=false.')
            checkpoint_kwargs['use_reentrant'] = False
            args.gradient_checkpointing_kwargs = checkpoint_kwargs
            logger.info('softdtw_distill set gradient checkpointing use_reentrant=false to preserve attention gradients.')

    def _student_tokenizer(self):
        processor = getattr(self.trainer_ref, 'processing_class', None)
        if processor is None:
            processor = getattr(self.trainer_ref, 'tokenizer', None)
        return _get_tokenizer(processor)

    def _student_layers(self):
        from swift.model import get_llm_model

        model = getattr(self.trainer_ref, 'model', None)
        if model is None:
            return []
        try:
            return get_llm_model(model, inner_backbone=True).layers
        except (AttributeError, TypeError):
            return []

    def _student_attention(self, sample_index: int) -> Optional[torch.Tensor]:
        maps = []
        layers = self._student_layers()
        for layer_index in self.layers:
            if layer_index >= len(layers):
                continue
            weights = getattr(layers[layer_index].self_attn, '_applicable_attention_weights', None)
            if weights is not None:
                maps.append(weights[sample_index].float().mean(dim=0))
        if not maps:
            return None
        return torch.stack(maps).mean(dim=0)

    def _clear_student_attention(self):
        for layer in self._student_layers():
            if hasattr(layer.self_attn, '_applicable_attention_weights'):
                layer.self_attn._applicable_attention_weights = None

    @staticmethod
    def _ce_loss(outputs, labels, num_items_in_batch):
        from swift.trainers import per_token_loss_func

        token_loss = outputs.loss if getattr(outputs, 'loss', None) is not None else per_token_loss_func(outputs, labels)
        if num_items_in_batch is None:
            num_items_in_batch = (labels[:, 1:] != -100).sum()
        return token_loss.sum() / num_items_in_batch.clamp_min(1)

    def _pairwise_cost(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        if self.cost_metric == 'l2':
            return (student[:, None, :] - teacher[None, :, :]).square().sum(dim=-1)
        epsilon = torch.finfo(student.dtype).eps
        student = student.clamp_min(epsilon)
        teacher = teacher.clamp_min(epsilon)
        return ((student * student.log()).sum(dim=-1, keepdim=True) - student @ teacher.log().T).clamp_min(0)

    def _prepare_samples(self, input_ids: torch.Tensor):
        tokenizer = self._student_tokenizer()
        try:
            audio_token_id = self.trainer_ref.model.config.thinker_config.audio_token_id
        except AttributeError:
            audio_token_id = 151676

        samples = []
        for sample_index, row in enumerate(input_ids):
            audio_positions = (row == audio_token_id).nonzero(as_tuple=False).flatten()
            if audio_positions.numel() == 0:
                continue
            context, transcript = _extract_context_transcript(row, tokenizer)
            if not context or not transcript:
                continue
            context_ids = tokenizer.encode(context, add_special_tokens=False)
            context_positions = _find_subsequence(row.tolist(), context_ids)
            teacher_item = _build_teacher_ids(context, transcript, self.teacher.tokenizer)
            student_attention = self._student_attention(sample_index)
            if context_positions is None or teacher_item is None or student_attention is None:
                continue
            context_tensor = torch.tensor(context_positions, dtype=torch.long, device=row.device)
            student_map = student_attention[audio_positions][:, context_tensor]
            if min(student_map.shape) == 0:
                continue
            student_map = student_map / student_map.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            samples.append({'student': student_map, **teacher_item})
        return samples

    def _distill_loss(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        self.teacher.ensure_loaded(input_ids.device)
        samples = self._prepare_samples(input_ids)
        if not samples:
            return None
        teacher_ids, teacher_mask = _pad_batch(
            [sample['ids'] for sample in samples], self.teacher.pad_token_id, input_ids.device)
        teacher_output = self.teacher.forward(teacher_ids, teacher_mask)

        distances = []
        for batch_index, sample in enumerate(samples):
            layer_maps = []
            for layer_index in self.teacher_layers:
                if layer_index < len(teacher_output.attentions):
                    weights = teacher_output.attentions[layer_index]
                    if weights is not None:
                        layer_maps.append(weights[batch_index].float().mean(dim=0))
            if not layer_maps:
                continue
            teacher_attention = torch.stack(layer_maps).mean(dim=0)
            transcript_positions = torch.tensor(
                sample['transcript_positions'], dtype=torch.long, device=input_ids.device)
            context_positions = torch.tensor(sample['context_positions'], dtype=torch.long, device=input_ids.device)
            teacher_map = teacher_attention[transcript_positions][:, context_positions]
            teacher_map = teacher_map / teacher_map.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            context_width = min(sample['student'].shape[-1], teacher_map.shape[-1])
            cost = self._pairwise_cost(sample['student'][:, :context_width], teacher_map[:, :context_width])
            distance = soft_dtw(cost, self.gamma)
            if torch.isfinite(distance):
                distances.append(distance)
        return torch.stack(distances).mean() if distances else None

    def __call__(
        self,
        outputs,
        labels,
        *,
        num_items_in_batch=None,
        loss_scale=None,
        input_ids=None,
        **kwargs,
    ):
        ce_loss = self._ce_loss(outputs, labels, num_items_in_batch)
        if self.alpha <= 0:
            self._clear_student_attention()
            return ce_loss
        if input_ids is None:
            raise RuntimeError('softdtw_distill requires input_ids from Seq2SeqTrainer.compute_loss.')
        try:
            distill_loss = self._distill_loss(input_ids)
            if distill_loss is None:
                raise RuntimeError(
                    'No distillation samples were produced. Ensure the Qwen3-ASR attention plugin is loaded, '
                    'the configured layers are enabled, and the batch contains context, audio, and transcript tokens.')
            return ce_loss + self.alpha * distill_loss
        finally:
            self._clear_student_attention()
