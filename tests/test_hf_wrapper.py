import pytest

try:
    import transformers  # noqa: F401
except ImportError:
    pytest.skip("需要 transformers", allow_module_level=True)

import torch

from localsight.model import LocalsightHFConfig, LocalsightHFForCausalLM


def test_hf_wrapper_save_load(tmp_path):
    torch.manual_seed(0)
    model = LocalsightHFForCausalLM(LocalsightHFConfig())
    model.save_pretrained(tmp_path)
    loaded = LocalsightHFForCausalLM.from_pretrained(tmp_path)

    for (_, p1), (_, p2) in zip(model.core.named_parameters(), loaded.core.named_parameters()):
        assert torch.equal(p1, p2)

    ids = torch.randint(0, 6400, (1, 8))
    out = model(ids, labels=ids)
    assert out.loss is not None
    assert torch.isfinite(out.loss)
