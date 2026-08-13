import torch

from localsight.data.packing import pack_sequences


def test_packing_shapes_and_separators():
    docs = [[1, 2, 3], [4, 5], [6]]
    batches = list(pack_sequences(docs, max_len=9, eos_id=0, pad_id=-100))
    assert len(batches) == 1
    ids, dids = batches[0]
    assert ids.shape == (1, 9)
    assert dids.shape == (1, 9)
    assert ids[0].tolist() == [1, 2, 3, 0, 4, 5, 0, 6, 0]
    assert dids[0].tolist() == [0, 0, 0, 0, 1, 1, 1, 2, 2]


def test_packing_continues_across_batches():
    docs = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    batches = list(pack_sequences(docs, max_len=6, eos_id=0, pad_id=-100))
    assert len(batches) == 2
    first_ids, first_dids = batches[0]
    # 前 5 个 token + eos 恰好填满 6
    assert first_ids[0].tolist() == [1, 2, 3, 4, 5, 0]
    assert (first_dids[0] == 0).all()
    second_ids, second_dids = batches[1]
    assert second_ids[0, :6].tolist() == [6, 7, 8, 9, 10, 0]
    assert (second_dids[0, :6] == 1).all()


def test_document_ids_increment():
    docs = [[1], [2], [3]]
    _, dids = next(pack_sequences(docs, max_len=12, eos_id=0, pad_id=-100))
    assert dids[0, :6].tolist() == [0, 0, 1, 1, 2, 2]
