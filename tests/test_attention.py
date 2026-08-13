import torch

from localsight.model import LocalsightConfig, LocalsightModel, KVCache
from localsight.model.attention import build_document_causal_mask


def test_document_causal_mask_structure():
    doc = torch.tensor([[0, 0, 1, 1]])
    mask = build_document_causal_mask(doc)
    assert mask.shape == (1, 1, 4, 4)
    assert mask.dtype == torch.bool
    assert not mask[0, 0, 2, 0]
    assert not mask[0, 0, 2, 1]
    assert mask[0, 0, 2, 2]
    assert not mask[0, 0, 2, 3]  # 因果：不能看未来
    assert mask[0, 0, 3, 2]
    assert mask[0, 0, 3, 3]
    assert not mask[0, 0, 3, 0]


def test_future_tokens_do_not_leak():
    torch.set_num_threads(1)  # 避免 BLAS 多线程浮点非确定性干扰断言
    torch.manual_seed(0)
    cfg = LocalsightConfig()
    model = LocalsightModel(cfg)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        out1, _ = model(ids)
        ids2 = ids.clone()
        ids2[0, 8] = 777
        out2, _ = model(ids2)
    # 因果性成立；GEMM 分块差异会产生 ~1e-8 级浮点伪差，这里只要求“无有意义泄漏”
    assert (out1[:, :8] - out2[:, :8]).abs().max().item() < 2e-5


def test_prefill_matches_incremental_decode():
    torch.manual_seed(1)
    cfg = LocalsightConfig()
    model = LocalsightModel(cfg)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        full, _ = model(ids)

        cache = KVCache(
            cfg.num_hidden_layers,
            1,
            cfg.num_key_value_heads,
            cfg.head_dim,
            32,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        model(ids[:, :8], cache=cache)
        cache.commit()
        last = None
        for i in range(8, 16):
            last, _ = model(ids[:, i:i + 1], cache=cache)
            cache.commit()
    assert last is not None
    torch.testing.assert_close(last, full[:, 15:16], rtol=1e-3, atol=1e-4)


def test_document_packing_blocks_cross_doc_attention():
    torch.set_num_threads(1)
    torch.manual_seed(2)
    cfg = LocalsightConfig()
    model = LocalsightModel(cfg)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    docs = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
    with torch.no_grad():
        out, _ = model(ids, document_ids=docs)
        ids2 = ids.clone()
        ids2[0, 5] = 888  # 另一文档内
        out2, _ = model(ids2, document_ids=docs)
    # 文档 0 的 4 个位置不应受文档 1 变化影响
    assert (out[:, :4] - out2[:, :4]).abs().max().item() < 2e-5
