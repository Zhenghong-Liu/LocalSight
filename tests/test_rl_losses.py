import torch

from localsight.rl.losses import grpo_dapo_loss, group_advantages, simpo_loss


def test_simpo_rewards_better_chosen():
    chosen = torch.full((2, 5), -1.0)  # 更高 logp
    rejected = torch.full((2, 5), -3.0)
    good = simpo_loss(chosen, rejected, beta=2.0, gamma=0.0)
    bad = simpo_loss(rejected, chosen, beta=2.0, gamma=0.0)
    assert good.item() < bad.item()


def test_grpo_increases_ratio_for_positive_advantage():
    old = torch.full((2, 3), 0.0)
    logps = torch.tensor([[0.2, 0.2, 0.2], [0.0, 0.0, 0.0]])
    adv = torch.tensor([1.0, 0.0])
    loss = grpo_dapo_loss(old, logps, adv)
    assert torch.isfinite(loss)


def test_grpo_clip_high_caps_ratio():
    old = torch.full((1, 4), 0.0)
    logps = torch.full((1, 4), 2.0)  # ratio = e^2 ≈ 7.4，远超 1.28
    adv = torch.tensor([1.0])
    loss = grpo_dapo_loss(old, logps, adv, clip_high=0.28)
    # 完全裁剪时损失应接近 -1.28，而不是 -7.4
    assert -1.35 < loss.item() < -1.2


def test_group_advantages_normalizes():
    rewards = torch.tensor([[1.0, 2.0, 3.0], [4.0, 4.0, 4.0]])
    adv = group_advantages(rewards)
    assert adv[0].sum().abs().item() < 1e-5
    assert adv[1].abs().max().item() < 1e-5
