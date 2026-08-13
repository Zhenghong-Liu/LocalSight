from localsight.data.minhash import MinHashSketch, deduplicate


def test_identical_texts_have_jaccard_one():
    a = MinHashSketch("今天天气很好，我们去公园散步。")
    b = MinHashSketch("今天天气很好，我们去公园散步。")
    assert a.jaccard(b) == 1.0


def test_different_texts_have_low_jaccard():
    a = MinHashSketch("今天天气很好，我们去公园散步。" * 4)
    b = MinHashSketch("机器学习模型的训练需要大量高质量数据。" * 4)
    assert a.jaccard(b) < 0.3


def test_deduplicate_keeps_one_of_pair():
    rows = [(0, "完全相同的文本内容。" * 5), (1, "完全相同的文本内容。" * 5)]
    kept = deduplicate(rows, threshold=0.8)
    assert kept == [0]
