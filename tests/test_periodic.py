import torch

from localsight.eval.periodic import generate_padded_batch, rep_n
from localsight.model import LocalsightConfig, LocalsightForCausalLM


class _FakeTokenizer:
    def decode(self, ids: list[int]) -> str:
        return ",".join(str(i) for i in ids)


def test_rep_n_basics():
    assert rep_n("", 4) == 0.0
    assert rep_n("abc", 4) == 0.0
    assert rep_n("abcdabcdabcd", 4) == 1.0
    assert rep_n("独一无二的答案", 4) == 0.0
    assert 0.0 < rep_n("你好你好你好你好", 2) <= 1.0


def test_generate_padded_batch_shapes_and_decoding():
    torch.manual_seed(0)
    cfg = LocalsightConfig()
    model = LocalsightForCausalLM(cfg).eval()
    tok = _FakeTokenizer()
    prompts = [[5, 6, 7], [11, 12, 13, 14, 15]]
    texts = generate_padded_batch(
        model, tok, prompts, max_new=8, temperature=1.0, top_p=0.9, stop_id=0,
        model_cfg=cfg, device=torch.device("cpu"), dtype=torch.float32,
    )
    assert len(texts) == 2
    for text in texts:
        assert isinstance(text, str)
        assert text != ""


def test_generate_padded_batch_greedy_runs():
    torch.manual_seed(1)
    cfg = LocalsightConfig()
    model = LocalsightForCausalLM(cfg).eval()
    tok = _FakeTokenizer()
    texts = generate_padded_batch(
        model, tok, [[3, 4, 5]], max_new=4, temperature=0.0, top_p=1.0, stop_id=-1,
        model_cfg=cfg, device=torch.device("cpu"), dtype=torch.float32,
    )
    assert len(texts) == 1
    assert len(texts[0].split(",")) == 4
