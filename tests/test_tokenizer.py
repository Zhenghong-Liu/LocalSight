from pathlib import Path

from localsight.tokenizer.loader import LocalSightTokenizer

TOKENIZER_DIR = Path(__file__).resolve().parents[1] / "data" / "tokenizer"


def test_tokenizer_snapshot():
    tok = LocalSightTokenizer(TOKENIZER_DIR)
    assert tok.vocab_size == 6400
    assert tok.eos_id == 0
    assert tok.think_start_id >= 0
    assert tok.think_end_id >= 0
    ids = tok.encode("你好世界")
    assert tok.decode(ids) == "你好世界"
