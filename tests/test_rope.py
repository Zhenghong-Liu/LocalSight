import torch

from localsight.model import apply_rotary_emb, build_rope_cache, rotate_half


def test_apply_matches_rotate_half_formula():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 16)
    cos, sin = build_rope_cache(4, 16, base=1e6)
    out = apply_rotary_emb(x, cos, sin)
    x1, x2 = x.chunk(2, dim=-1)
    ref = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-7)


def test_position_zero_is_identity():
    x = torch.randn(2, 1, 8)
    cos, sin = build_rope_cache(1, 8, base=1e6)
    torch.testing.assert_close(apply_rotary_emb(x, cos, sin), x)


def test_rotation_preserves_pair_norm():
    torch.manual_seed(2)
    x = torch.randn(3, 6, 32)
    cos, sin = build_rope_cache(6, 32, base=1e6)
    out = apply_rotary_emb(x, cos, sin)
    assert torch.allclose(out.pow(2).sum(-1), x.pow(2).sum(-1), atol=1e-4)


def test_different_positions_differ():
    torch.manual_seed(3)
    x = torch.randn(1, 1, 8)
    cos, sin = build_rope_cache(8, 8, base=1e6)
    y2 = apply_rotary_emb(x, cos[2:3], sin[2:3])
    y5 = apply_rotary_emb(x, cos[5:6], sin[5:6])
    assert not torch.allclose(y2, y5)


def test_yarn_scaling_changes_frequencies():
    a = build_rope_cache(8, 8, base=1e6)[0]
    b = build_rope_cache(8, 8, base=1e6, rope_scaling={"type": "yarn", "factor": 4.0})[0]
    assert not torch.allclose(a, b)
