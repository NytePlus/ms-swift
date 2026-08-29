import importlib.util
from pathlib import Path

import torch


PLUGIN = Path(__file__).parents[2] / 'examples' / 'custom' / 'qwen3_asr_distill_plugin.py'
SPEC = importlib.util.spec_from_file_location('qwen3_asr_distill_plugin_test', PLUGIN)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DummyAttention(torch.nn.Module):
    num_key_value_groups = 1

    def __init__(self, enabled=False, transform='identity'):
        super().__init__()
        self._applicable_attention_enabled = enabled
        self._applicable_attention_capture = True
        self._applicable_attention_config = {'transform': transform, 'scale': 0.5}


def _official_eager(module, query, key, value, attention_mask, scaling):
    logits = query @ key.transpose(2, 3) * scaling
    if attention_mask is not None:
        logits = logits + attention_mask
    weights = torch.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
    return (weights @ value).transpose(1, 2).contiguous(), weights


def test_identity_matches_eager_output_and_gradient():
    torch.manual_seed(7)
    query = torch.randn(2, 3, 4, 5, requires_grad=True)
    key = torch.randn(2, 3, 4, 5, requires_grad=True)
    value = torch.randn(2, 3, 4, 5, requires_grad=True)
    mask = torch.zeros(2, 1, 4, 4)
    mask[:, :, :, -1] = torch.finfo(query.dtype).min

    expected, _ = _official_eager(DummyAttention(), query, key, value, mask, 5**-0.5)
    expected.sum().backward()
    expected_grad = query.grad.clone()
    query.grad = None

    actual, weights = MODULE.applicable_attention_forward(
        DummyAttention(enabled=True), query, key, value, mask, scaling=5**-0.5)
    actual.sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(query.grad, expected_grad)
    assert weights.requires_grad


def test_transform_is_applied_before_softmax():
    query = torch.tensor([[[[1.0, 0.0]]]], requires_grad=True)
    key = torch.tensor([[[[2.0, 0.0], [0.0, 1.0]]]])
    value = torch.eye(2).reshape(1, 1, 2, 2)
    _, weights = MODULE.applicable_attention_forward(
        DummyAttention(enabled=True, transform='scale'), query, key, value, None, scaling=1.0)
    expected = torch.softmax(torch.tensor([1.0, 0.0]), dim=-1)
    torch.testing.assert_close(weights.flatten(), expected)
    weights.sum().backward()
    assert query.grad is not None
