import math
import os

from localsight.training.pretrain import cosine_resume_schedule, interrupted_lr


def test_cosine_resume_schedule_boundaries():
    peak = 2.96e-3
    ratio = 0.1
    assert abs(cosine_resume_schedule(1000, 1000, 5500, peak, ratio) - peak) < 1e-12
    assert abs(cosine_resume_schedule(5500, 1000, 5500, peak, ratio) - peak * ratio) < 1e-12
    mid = cosine_resume_schedule(3250, 1000, 5500, peak, ratio)
    assert peak * ratio < mid < peak


def test_cosine_resume_schedule_monotonic():
    vals = [cosine_resume_schedule(s, 1000, 5500, 3e-3, 0.1) for s in range(1000, 5501, 500)]
    assert all(b <= a + 1e-12 for a, b in zip(vals, vals[1:]))


def test_interrupted_lr_close_to_peak():
    os.environ["WORLD_SIZE"] = "2"
    cfg = {
        "micro_batch_size": 32,
        "epochs": 5,
        "grad_accum": 4,
        "warmup_ratio": 0.02,
        "lr": 3e-3,
        "lr_min_ratio": 0.1,
    }
    lr = interrupted_lr(cfg, dataset_len=520_456, val_sequences=200, step=1000)
    assert 2.7e-3 < lr <= 3.0e-3
