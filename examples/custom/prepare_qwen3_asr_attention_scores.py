#!/usr/bin/env python3
"""Prepare full-history predictions for the repository's exact legacy scorer."""

import argparse
import json
from pathlib import Path


DEFAULT_GROUPS = (
    'baseline',
    'correct-a025',
    'correct-a050',
    'correct-a100',
    'correct-a200',
    'shuffled-a025',
    'shuffled-a050',
    'shuffled-a100',
    'shuffled-a200',
    'zero-context',
    'matched-random',
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def read_predictions(paths: list[Path]) -> dict[str, str]:
    predictions = {}
    for path in paths:
        with path.open(encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                row = line.rstrip('\n').split(maxsplit=1)
                if not row:
                    continue
                key, text = row[0], row[1] if len(row) == 2 else ''
                if key in predictions:
                    raise ValueError(f'duplicate prediction key {key!r} in {path}:{line_number}')
                predictions[key] = text
    return predictions


def replace_symlink(path: Path, target: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.exists():
        path.unlink()
    path.symlink_to(target)


def prepare_group(
    group: str,
    prediction_root: Path,
    keys: list[str],
    compat_root: Path,
    scorer_dir: Path,
) -> dict:
    group_dir = prediction_root / group
    shard_paths = sorted(group_dir.glob('part-*.pred'))
    if len(shard_paths) != 8:
        raise ValueError(f'{group}: expected 8 shard files, found {len(shard_paths)}')
    predictions = read_predictions(shard_paths)
    expected_keys = set(keys)
    missing = expected_keys - predictions.keys()
    extra = predictions.keys() - expected_keys
    if missing or extra:
        raise ValueError(f'{group}: missing={len(missing)} extra={len(extra)}')

    merged = group_dir / 'merged.pred'
    with merged.open('w', encoding='utf-8') as f:
        for key in keys:
            f.write(f'{key} {predictions[key]}\n')

    pipeline_root = compat_root / group / 'pipeline'
    local_score = pipeline_root / 'score'
    local_score.mkdir(parents=True, exist_ok=True)
    for name in ('normalize.py', 'wenet_compute_cer.py', 'score_old.py'):
        replace_symlink(local_score / name, scorer_dir / name)
    expected_pred = local_score / 'results/qwen_asr_pred/dev/dev_qwen_asr_pred'
    replace_symlink(expected_pred, merged.resolve())

    return {
        'group': group,
        'merged_prediction': str(merged.resolve()),
        'pipeline_root': str(pipeline_root.resolve()),
        'result': str((Path(f'{expected_pred}_norm_wer')).resolve()),
        'num_predictions': len(predictions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction-root', type=Path, required=True)
    parser.add_argument('--history-manifest', type=Path, required=True)
    parser.add_argument('--hotword-manifest', type=Path, required=True)
    parser.add_argument('--compat-root', type=Path, required=True)
    parser.add_argument('--scorer-dir', type=Path, required=True)
    parser.add_argument('--groups', nargs='+', default=list(DEFAULT_GROUPS))
    args = parser.parse_args()

    history_rows = load_jsonl(args.history_manifest)
    hotword_rows = {row['key']: row for row in load_jsonl(args.hotword_manifest)}
    keys = [row['key'] for row in history_rows]
    if len(keys) != len(set(keys)):
        raise ValueError('history manifest contains duplicate keys')
    if set(keys) != set(hotword_rows):
        raise ValueError('history and hotword manifests do not contain the same utterance IDs')

    # score_old.sh hard-codes these dev paths.  They intentionally contain the
    # full test/history references while the unmodified host script is running.
    multitask_path = args.compat_root / 'home/data/slidespeech/dev_oracle_v1/multitask.jsonl'
    gt_path = args.compat_root / 'data/slidespeech/dev_oracle_v1/hotword/gt.txt'
    multitask_path.parent.mkdir(parents=True, exist_ok=True)
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    with multitask_path.open('w', encoding='utf-8') as multitask, gt_path.open('w', encoding='utf-8') as gt:
        for row in history_rows:
            scored = dict(row)
            scored['hotword'] = hotword_rows[row['key']]['hotword']
            multitask.write(json.dumps(scored, ensure_ascii=False) + '\n')
            gt.write(f"{row['key']} {row['target']}\n")

    prepared = [
        prepare_group(group, args.prediction_root, keys, args.compat_root, args.scorer_dir)
        for group in args.groups
    ]
    manifest = args.compat_root / 'prepared_scores.json'
    manifest.write_text(json.dumps(prepared, indent=2) + '\n', encoding='utf-8')
    print(manifest)


if __name__ == '__main__':
    main()
