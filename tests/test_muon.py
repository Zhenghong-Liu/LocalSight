import torch

from localsight.training.muon import Muon, zeropower_via_newtonschulz5


def test_newton_schulz_output_is_orthonormal_columns():
    torch.manual_seed(0)
    g = torch.randn(64, 16)
    o = zeropower_via_newtonschulz5(g, steps=5)
    gram = o.float().T @ o.float()
    torch.testing.assert_close(gram, torch.eye(16), atol=1e-3, rtol=1e-3)


def test_muon_updates_matrix_and_vector_params():
    torch.manual_seed(1)
    matrix = torch.nn.Parameter(torch.randn(8, 4))
    vector = torch.nn.Parameter(torch.randn(4))
    opt = Muon([matrix, vector], lr=1e-2, wd=0.0)
    (matrix.sum() + vector.sum()).backward()
    m0 = matrix.clone()
    v0 = vector.clone()
    opt.step()
    assert not torch.equal(matrix, m0)
    assert not torch.equal(vector, v0)


def test_muon_minimizes_quadratic():
    torch.manual_seed(2)
    target = torch.randn(4, 4)
    x = torch.nn.Parameter(torch.randn(4, 4))
    opt = Muon([x], lr=5e-2, wd=0.0, muon_scale=0.5)
    loss0 = None
    for _ in range(200):
        loss = ((x - target) ** 2).mean()
        if loss0 is None:
            loss0 = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert ((x - target) ** 2).mean().item() < loss0


def test_weight_decay_shrinks_weights():
    torch.manual_seed(3)
    x = torch.nn.Parameter(torch.ones(2, 2))
    opt = Muon([x], lr=1e-2, wd=0.5)
    x.grad = torch.zeros_like(x)
    opt.step()
    assert x.abs().max().item() < 1.0
