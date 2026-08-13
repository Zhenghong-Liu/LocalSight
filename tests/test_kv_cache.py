import torch

from localsight.model import KVCache


def _cache() -> KVCache:
    return KVCache(2, 1, 4, 8, 16, dtype=torch.float32, device=torch.device("cpu"))


def test_append_commit_and_read():
    cache = _cache()
    k0 = torch.randn(1, 5, 4, 8)
    v0 = torch.randn(1, 5, 4, 8)
    k1 = torch.randn(1, 5, 4, 8)
    v1 = torch.randn(1, 5, 4, 8)
    cache.append(0, k0, v0)
    cache.append(1, k1, v1)
    cache.commit()
    assert cache.length == 5
    assert torch.equal(cache.k(0), k0)
    assert torch.equal(cache.v(1), v1)


def test_spawn_copies_prefix():
    cache = _cache()
    k0 = torch.randn(1, 4, 4, 8)
    v0 = torch.randn(1, 4, 4, 8)
    k1 = torch.randn(1, 4, 4, 8)
    v1 = torch.randn(1, 4, 4, 8)
    cache.append(0, k0, v0)
    cache.append(1, k1, v1)
    cache.commit()
    children = cache.spawn(3)
    assert len(children) == 3
    for child in children:
        assert child.length == 4
        assert torch.equal(child.k(0), k0)
        assert torch.equal(child.v(1), v1)


def test_overflow_raises():
    cache = _cache()
    k = torch.randn(1, 17, 4, 8)
    try:
        cache.append(0, k, k)
    except RuntimeError:
        return
    raise AssertionError("溢出应该抛 RuntimeError")
