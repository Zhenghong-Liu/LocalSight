import torch

from localsight.model import RMSNorm


def _reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    return (y * weight.float()).to(x.dtype)


def test_rmsnorm_matches_reference():
    torch.manual_seed(0)
    x = torch.randn(4, 8, 64)
    weight = torch.randn(64)
    layer = RMSNorm(64, eps=1e-6)
    with torch.no_grad():
        layer.weight.copy_(weight)
    torch.testing.assert_close(layer(x), _reference(x, weight, 1e-6), rtol=1e-5, atol=1e-6)


def test_rmsnorm_preserves_dtype():
    x = torch.randn(2, 16, dtype=torch.bfloat16)
    assert RMSNorm(16)(x).dtype == torch.bfloat16


def test_rmsnorm_unit_variance():
    torch.manual_seed(1)
    x = torch.randn(3, 32)
    out = RMSNorm(32, eps=1e-6)(x)
    var = out.float().pow(2).mean(-1)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-5)
