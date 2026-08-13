import json
from pathlib import Path

from localsight.data.dataset import PretrainDataset
from localsight.data.pretrain import PretrainDataBuilder
from localsight.tokenizer.loader import LocalSightTokenizer

TOKENIZER_DIR = Path(__file__).resolve().parents[1] / "data" / "tokenizer"


def test_build_and_load(tmp_path: Path):
    tok = LocalSightTokenizer(TOKENIZER_DIR)
    src = tmp_path / "pretrain.jsonl"
    docs = [
        "今天天气很好。",
        "今天天气很好。",  # 精确重复
        "机器学习模型需要大量高质量的训练数据。" * 5,
        "短",  # 过短，应被过滤
    ]
    src.write_text(
        "\n".join(json.dumps({"text": t}, ensure_ascii=False) for t in docs),
        encoding="utf-8",
    )

    builder = PretrainDataBuilder(tok, max_len=64, min_chars=3)
    manifest = builder.build(src, tmp_path / "out")
    stats = manifest["stats"]
    assert stats["rows"] == 4
    assert stats["removed_too_short"] == 1
    assert stats["removed_exact_dup"] == 1
    assert stats["kept_rows"] == 2
    assert stats["sequences"] >= 1

    ds = PretrainDataset(tmp_path / "out")
    assert len(ds) == stats["sequences"]
    item = ds[0]
    assert item["input_ids"].shape == (64,)
    assert item["document_ids"].shape == (64,)
    pad = item["input_ids"] == -1
    assert bool((item["labels"][pad] == -100).all())
    assert bool((item["document_ids"][pad] == -1).all())
