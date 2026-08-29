# Copyright (c) ModelScope Contributors. All rights reserved.
"""Run causal test-time T<-C attention interventions on Qwen3-ASR.

The text-only teacher reuses the current Qwen3-ASR checkpoint's thinker text
backbone. It sees the same correct or token-shuffled history and generated
prefix as the student, but the audio span is removed. Teacher state is updated
after every generated token, so no future transcript token is used.
"""

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import kaldiio
import torch
from qwen_asr import Qwen3ASRModel
from transformers import StoppingCriteria, StoppingCriteriaList

os.environ.setdefault('QWEN3_ASR_ATTENTION_STANDALONE', '1')

from examples.custom.qwen3_asr_distill_plugin import (BACKEND_NAME, AttentionInterventionRuntime,
                                                      clear_attention_intervention_runtime,
                                                      set_attention_intervention_runtime)  # noqa: E402

logger = logging.getLogger(__name__)


def _find_subsequence(sequence: Sequence[int], subsequence: Sequence[int]) -> Tuple[int, int]:
    if not subsequence:
        return 0, 0
    width = len(subsequence)
    for start in range(len(sequence) - width + 1):
        if list(sequence[start:start + width]) == list(subsequence):
            return start, start + width
    raise ValueError('History token sequence was not found in the Qwen3-ASR prompt.')


def _left_pad(rows: List[torch.Tensor], pad_value: int, device):
    width = max(row.numel() for row in rows)
    values = torch.full((len(rows), width), pad_value, dtype=torch.long, device=device)
    mask = torch.zeros((len(rows), width), dtype=torch.bool, device=device)
    offsets = []
    for index, row in enumerate(rows):
        offset = width - row.numel()
        values[index, offset:] = row.to(device)
        mask[index, offset:] = True
        offsets.append(offset)
    return values, mask, offsets


def _build_teacher_inputs(asr, student_ids, student_attention_mask, contexts, shuffled, seed):
    tokenizer = asr.processor.tokenizer
    thinker_config = asr.model.config.thinker_config
    audio_ids = {
        thinker_config.audio_start_token_id,
        thinker_config.audio_end_token_id,
        thinker_config.audio_token_id,
    }
    teacher_rows = []
    teacher_context_rows = []
    student_context_mask = torch.zeros_like(student_attention_mask, dtype=torch.bool)

    for batch_index, context in enumerate(contexts):
        valid_positions = student_attention_mask[batch_index].bool().nonzero(as_tuple=False).flatten()
        unpadded = student_ids[batch_index].index_select(0, valid_positions)
        unpadded_list = unpadded.tolist()
        context_ids = tokenizer(context, add_special_tokens=False).input_ids if context else []
        context_start, context_end = _find_subsequence(unpadded_list, context_ids)
        if context_ids:
            padded_positions = valid_positions[context_start:context_end]
            student_context_mask[batch_index, padded_positions] = True

        keep = torch.tensor([token not in audio_ids for token in unpadded_list], device=unpadded.device)
        teacher_row = unpadded[keep]
        kept_original_positions = torch.arange(unpadded.numel(), device=unpadded.device)[keep]
        teacher_context = (kept_original_positions >= context_start) & (kept_original_positions < context_end)
        if shuffled and teacher_context.any():
            indices = teacher_context.nonzero(as_tuple=False).flatten()
            generator = torch.Generator(device=teacher_row.device)
            generator.manual_seed(seed + batch_index)
            permutation = torch.randperm(indices.numel(), generator=generator, device=teacher_row.device)
            shuffled_values = teacher_row.index_select(0, indices).index_select(0, permutation)
            teacher_row = teacher_row.clone()
            teacher_row.index_copy_(0, indices, shuffled_values)
        teacher_rows.append(teacher_row)
        teacher_context_rows.append(teacher_context)

    teacher_ids, teacher_attention_mask, offsets = _left_pad(
        teacher_rows, tokenizer.pad_token_id, student_ids.device)
    teacher_context_mask = torch.zeros_like(teacher_attention_mask)
    for batch_index, (context_row, offset) in enumerate(zip(teacher_context_rows, offsets)):
        teacher_context_mask[batch_index, offset:] = context_row
    return teacher_ids, teacher_attention_mask, teacher_context_mask, student_context_mask


class TeacherMapUpdater(StoppingCriteria):
    """Update prefix-only teacher attention after each generated student token."""

    def __init__(self, backbone, runtime, past_key_values, attention_mask, context_mask):
        self.backbone = backbone
        self.runtime = runtime
        self.past_key_values = past_key_values
        self.attention_mask = attention_mask
        self.context_mask = context_mask

    @torch.no_grad()
    def __call__(self, input_ids, scores, **kwargs):
        token = input_ids[:, -1:]
        self.attention_mask = torch.cat(
            [self.attention_mask, torch.ones_like(token, dtype=self.attention_mask.dtype)], dim=-1)
        position_ids = self.attention_mask.long().sum(dim=-1, keepdim=True) - 1
        self.runtime.begin_teacher(self.context_mask)
        outputs = self.backbone(
            input_ids=token,
            attention_mask=self.attention_mask,
            position_ids=position_ids,
            past_key_values=self.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        self.past_key_values = outputs.past_key_values
        self.runtime.begin_student(self.runtime.student_context_mask)
        self.runtime.advance()
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


class IntervenedQwen3ASRModel(Qwen3ASRModel):
    """Qwen3-ASR inference wrapper that drives the registered attention backend."""

    intervention_strategy = 'baseline'
    intervention_alpha = 1.0
    intervention_layers = None
    intervention_seed = 42

    def configure_intervention(self, strategy, alpha, layers, seed):
        self.intervention_strategy = strategy
        self.intervention_alpha = alpha
        self.intervention_layers = layers
        self.intervention_seed = seed
        self.intervention_stats = {
            'teacher_captures': 0,
            'applied_rows': 0,
            'max_abs_delta': 0.0,
            'context_mass_before': 0.0,
            'context_mass_after': 0.0,
        }
        self.model.thinker.model.config._attn_implementation = BACKEND_NAME
        selected = None if layers is None else set(layers)
        for index, layer in enumerate(self.model.thinker.model.layers):
            attention = layer.self_attn
            attention._applicable_attention_layer_index = index
            attention._applicable_attention_enabled = selected is None or index in selected
            attention._applicable_attention_capture = False
            attention._applicable_attention_config = {'transform': 'identity'}

    @torch.no_grad()
    def _infer_asr_transformers(self, contexts, wavs, languages):
        outputs = []
        texts = [self._build_text_prompt(context=context, force_language=language)
                 for context, language in zip(contexts, languages)]
        batch_size = self.max_inference_batch_size
        if batch_size is None or batch_size < 0:
            batch_size = len(texts)

        for start in range(0, len(texts), batch_size):
            sub_texts = texts[start:start + batch_size]
            sub_contexts = contexts[start:start + batch_size]
            sub_wavs = wavs[start:start + batch_size]
            inputs = self.processor(text=sub_texts, audio=sub_wavs, return_tensors='pt', padding=True)
            inputs = inputs.to(self.model.device).to(self.model.dtype)
            runtime = AttentionInterventionRuntime(
                self.intervention_strategy,
                alpha=self.intervention_alpha,
                layers=self.intervention_layers,
                seed=self.intervention_seed + start,
            )
            set_attention_intervention_runtime(runtime)
            stopping_criteria = None
            try:
                teacher_ids, teacher_attention_mask, teacher_context_mask, student_context_mask = (
                    _build_teacher_inputs(
                        self,
                        inputs['input_ids'],
                        inputs['attention_mask'],
                        sub_contexts,
                        runtime.strategy == 'shuffled_teacher',
                        runtime.seed,
                    ))
                runtime.begin_student(student_context_mask)
                if runtime.needs_teacher:
                    runtime.begin_teacher(teacher_context_mask)
                    teacher_position_ids = teacher_attention_mask.long().cumsum(dim=-1) - 1
                    teacher_position_ids.masked_fill_(~teacher_attention_mask, 0)
                    teacher_outputs = self.model.thinker.model(
                        input_ids=teacher_ids,
                        attention_mask=teacher_attention_mask,
                        position_ids=teacher_position_ids,
                        use_cache=True,
                        return_dict=True,
                    )
                    runtime.begin_student(student_context_mask)
                    stopping_criteria = StoppingCriteriaList([
                        TeacherMapUpdater(
                            self.model.thinker.model,
                            runtime,
                            teacher_outputs.past_key_values,
                            teacher_attention_mask,
                            teacher_context_mask,
                        )
                    ])
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    stopping_criteria=stopping_criteria,
                )
            finally:
                for name, value in runtime.stats.items():
                    if name == 'max_abs_delta':
                        self.intervention_stats[name] = max(self.intervention_stats[name], value)
                    else:
                        self.intervention_stats[name] += value
                clear_attention_intervention_runtime()
            decoded = self.processor.batch_decode(
                generated.sequences[:, inputs['input_ids'].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            outputs.extend(decoded)
        return outputs


def _read_items(path: Path, shard_index: int, num_shards: int, limit: int):
    items = []
    with path.open(encoding='utf-8') as stream:
        for index, line in enumerate(stream):
            if index % num_shards != shard_index:
                continue
            items.append(json.loads(line))
            if limit and len(items) >= limit:
                break
    return items


def _processed_ids(path: Path):
    if not path.exists():
        return set()
    with path.open(encoding='utf-8') as stream:
        return {line.split(maxsplit=1)[0] for line in stream if line.strip()}


def _audio(item, old_prefix: str, new_prefix: str):
    path = item['path'].replace(old_prefix, new_prefix, 1)
    sample_rate, waveform = kaldiio.load_mat(path)
    return waveform, sample_rate


def _batches(items: Sequence[Dict], batch_size: int) -> Iterable[List[Dict]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start:start + batch_size])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--strategy', choices=sorted(AttentionInterventionRuntime.STRATEGIES), required=True)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--layers', default='14,21,27')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-new-tokens', type=int, default=192)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--path-prefix-from', default='/aistor/sjtu/hpc_stor01/home/wangchencheng/data/slidespeech')
    parser.add_argument('--path-prefix-to', required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    layers = [int(value) for value in args.layers.split(',') if value.strip()] if args.layers else None
    items = _read_items(args.manifest, args.shard_index, args.num_shards, args.limit)
    done = _processed_ids(args.output)
    items = [item for item in items if item['key'] not in done]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logger.info('Loading %s; strategy=%s alpha=%s layers=%s shard=%s/%s remaining=%s',
                args.model_path, args.strategy, args.alpha, layers, args.shard_index, args.num_shards, len(items))
    started = time.time()
    model = IntervenedQwen3ASRModel.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map='cuda:0',
        attn_implementation='eager',
        max_inference_batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    model.configure_intervention(args.strategy, args.alpha, layers, args.seed + args.shard_index * 100000)

    completed = 0
    with args.output.open('a', encoding='utf-8') as output_stream:
        for batch in _batches(items, args.batch_size):
            audios = [_audio(item, args.path_prefix_from, args.path_prefix_to) for item in batch]
            contexts = [item.get('history', '') for item in batch]
            results = model.transcribe(audio=audios, context=contexts, language='English')
            for item, result in zip(batch, results):
                output_stream.write(f"{item['key']} {result.text}\n")
            output_stream.flush()
            completed += len(batch)
            logger.info('completed=%s/%s elapsed=%.1fs', completed, len(items), time.time() - started)

    summary = {
        'strategy': args.strategy,
        'alpha': args.alpha,
        'layers': layers,
        'shard_index': args.shard_index,
        'num_shards': args.num_shards,
        'completed': completed,
        'elapsed_seconds': time.time() - started,
        'intervention_stats': model.intervention_stats,
    }
    args.output.with_suffix(args.output.suffix + '.summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
