import torch

from swift.loss.mapping import loss_map
from swift.loss.softdtw_distill import soft_dtw


def test_softdtw_is_registered():
    assert loss_map['softdtw_distill'].__name__ == 'SoftDTWDistillLoss'


def test_softdtw_has_student_gradient():
    student = torch.tensor([[0.8, 0.2], [0.3, 0.7]], requires_grad=True)
    teacher = torch.tensor([[0.7, 0.3], [0.2, 0.8]])
    cost = (student[:, None, :] - teacher[None, :, :]).square().sum(dim=-1)
    loss = soft_dtw(cost, gamma=0.5)
    loss.backward()
    assert student.grad is not None
    assert torch.count_nonzero(student.grad) > 0
    assert torch.isfinite(student.grad).all()
