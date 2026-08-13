import torch

from localsight.model import LocalsightConfig, MoEFeedForward, MoEGate, load_balance_loss


def test_moe_forward_shape_and_top1():
    torch.manual_seed(0)
    cfg = LocalsightConfig()
    moe = MoEFeedForward(cfg)
    x = torch.randn(3, 7, cfg.hidden_size)
    out, aux = moe(x)
    assert out.shape == x.shape
    assert int(aux["expert_counts"].sum()) == 3 * 7
    assert aux["z_loss"] > 0
    assert aux["balance_loss"] >= 0


def test_balance_bias_updates_toward_uniform():
    torch.manual_seed(1)
    cfg = LocalsightConfig()
    gate = MoEGate(cfg)
    with torch.no_grad():
        gate.weight[0] = 100.0
        gate.weight[1:] = 0.0
        gate.bias.zero_()
    x = torch.rand(16, cfg.hidden_size) + 0.5  # 保证 logits 行 0 恒为正
    topk_idx, _, aux = gate(x)
    assert bool((topk_idx[:, 0] == 0).all())

    b0_before = gate.bias[0].item()
    gate.update_balance_bias(aux["expert_counts"], gamma=1e-3)
    assert gate.bias[0].item() < b0_before  # 超载专家降偏置
    assert gate.bias[1].item() > 0.0  # 欠载专家升偏置


def test_load_balance_loss_matches_manual():
    probs = torch.tensor([[0.7, 0.3], [0.7, 0.3], [0.2, 0.8]])
    idx = torch.tensor([[0], [0], [1]])
    loss = load_balance_loss(probs, idx, 2)
    expected = 2 * ((1.6 / 3) * (2 / 3) + (1.4 / 3) * (1 / 3))
    assert torch.allclose(loss, torch.tensor(expected), atol=1e-6)


def test_expert_specialization_shapes():
    torch.manual_seed(2)
    cfg = LocalsightConfig()
    moe = MoEFeedForward(cfg)
    # 手动构造：token 只流向 0 号专家时，输出等于专家输出
    x = torch.rand(2, 3, cfg.hidden_size) + 0.5
    with torch.no_grad():
        moe.gate.weight.zero_()
        moe.gate.weight[0, :] = 100.0
        moe.gate.bias.zero_()
    out, aux = moe(x)
    with torch.no_grad():
        ref = moe.experts[0](x)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-6)
