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
        self._applicable_attention_layer_index = 0
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


def test_zero_context_removes_only_context_mass():
    runtime = MODULE.AttentionInterventionRuntime('zero_context', layers=[0])
    runtime.begin_student(torch.tensor([[True, False]]))
    MODULE.set_attention_intervention_runtime(runtime)
    try:
        query = torch.tensor([[[[1.0, 0.0]]]])
        key = torch.tensor([[[[2.0, 0.0], [0.0, 1.0]]]])
        value = torch.eye(2).reshape(1, 1, 2, 2)
        _, weights = MODULE.applicable_attention_forward(
            DummyAttention(enabled=True), query, key, value, None, scaling=1.0)
    finally:
        MODULE.clear_attention_intervention_runtime()
    torch.testing.assert_close(weights.flatten(), torch.tensor([0.0, 1.0]))


def test_teacher_alpha_one_matches_teacher_context_mass():
    runtime = MODULE.AttentionInterventionRuntime('correct_teacher', alpha=1.0, layers=[0])
    runtime.teacher_context_mask = torch.tensor([[True, False]])
    runtime.student_context_mask = torch.tensor([[True, False, False]])
    runtime.teacher_weights[0] = torch.tensor([[[[0.75, 0.25]]]])
    student = torch.tensor([[[[0.2, 0.3, 0.5]]]])
    actual = MODULE._apply_attention_intervention(student, runtime, 0)
    torch.testing.assert_close(actual[..., 0], torch.tensor([[[0.75]]]))
    torch.testing.assert_close(actual.sum(dim=-1), torch.ones(1, 1, 1))


def test_matched_mass_random_preserves_context_mass():
    runtime = MODULE.AttentionInterventionRuntime('matched_mass_random', layers=[0], seed=7)
    runtime.begin_student(torch.tensor([[True, True, False]]))
    student = torch.tensor([[[[0.1, 0.3, 0.6]]]])
    actual = MODULE._apply_attention_intervention(student, runtime, 0)
    torch.testing.assert_close(actual[..., :2].sum(dim=-1), student[..., :2].sum(dim=-1))
    torch.testing.assert_close(actual[..., 2], student[..., 2])
