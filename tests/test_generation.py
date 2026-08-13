import torch

from localsight.generation import generate
from localsight.model import KVCache, LocalsightConfig, LocalsightForCausalLM


def _model() -> LocalsightForCausalLM:
    return LocalsightForCausalLM(LocalsightConfig()).eval()


def test_greedy_generation_matches_argmax():
    torch.manual_seed(0)
    model = _model()
    ids = torch.tensor([[5, 6, 7]])
    with torch.no_grad():
        logits, _, _ = model(ids)
        target = logits[:, -1].argmax(dim=-1)
    gen = generate(model, ids, max_new_tokens=1, temperature=0)
    assert gen[0, -1].item() == target.item()


def test_generation_with_cache_shapes():
    torch.manual_seed(1)
    model = _model()
    ids = torch.tensor([[5, 6, 7]])
    cache = KVCache(8, 1, 4, 96, 64, dtype=torch.float32, device=torch.device("cpu"))
    gen = generate(model, ids, max_new_tokens=6, temperature=1.0, top_p=0.9, cache=cache)
    assert gen.shape == (1, 9)
    assert bool((gen >= 0).all())
    assert bool((gen < 6400).all())
    assert cache.length == 9


def test_sampling_terminates_at_eos():
    torch.manual_seed(2)
    model = _model()
    ids = torch.tensor([[5, 6, 7]])
    gen = generate(model, ids, max_new_tokens=32, temperature=1.0, top_p=1.0, eos_id=0)
    assert gen.shape[1] <= ids.shape[1] + 32
