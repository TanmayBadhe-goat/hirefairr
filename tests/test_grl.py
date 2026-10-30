import torch

from hirefair.models.grl import GradientReversal


def test_forward_is_identity():
    grl = GradientReversal(3.0)
    x = torch.randn(4, 7)
    assert torch.equal(grl(x), x)


def test_backward_multiplies_by_minus_lambda():
    grl = GradientReversal(2.5)
    x = torch.ones(3, 5, requires_grad=True)
    grl(x).sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x, -2.5))


def test_set_lambda_changes_backward():
    grl = GradientReversal(1.0)
    grl.set_lambda(0.25)
    x = torch.ones(2, 2, requires_grad=True)
    grl(x).sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x, -0.25))


def test_lambda_zero_blocks_the_reversed_gradient():
    grl = GradientReversal(0.0)
    x = torch.ones(2, 2, requires_grad=True)
    grl(x).sum().backward()
    assert torch.allclose(x.grad, torch.zeros_like(x))
