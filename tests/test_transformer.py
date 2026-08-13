import torch

from localsight.model import LocalsightConfig, LocalsightForCausalLM, count_parameters


def test_exact_parameter_counts():
    model = LocalsightForCausalLM(LocalsightConfig())
    total, active = count_parameters(model)
    assert total == 198_416_640
    assert active == 63_936_768


def test_tied_embeddings():
    model = LocalsightForCausalLM(LocalsightConfig())
    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_forward_and_loss_decreases():
    torch.manual_seed(0)
    model = LocalsightForCausalLM(LocalsightConfig())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ids = torch.randint(0, 6400, (2, 16))
    logits, loss0, aux = model(ids, labels=ids)
    assert logits.shape == (2, 16, 6400)
    assert aux["z_loss"] > 0
    assert aux["expert_counts"].shape == (8, 4)
    loss0.backward()
    optimizer.step()
    optimizer.zero_grad()
    with torch.no_grad():
        _, loss1, _ = model(ids, labels=ids)
    assert loss1.item() < loss0.item()


def test_no_bias_and_scaled_output_init():
    model = LocalsightForCausalLM(LocalsightConfig())
    for name, param in model.named_parameters():
        assert not name.endswith(".bias"), f"架构要求无 bias: {name}"
    out_std = model.model.layers[0].self_attn.o_proj.weight.std().item()
    assert 0.003 < out_std < 0.007  # ~0.02/sqrt(16) = 0.005
