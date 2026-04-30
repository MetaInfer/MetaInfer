from __future__ import annotations

from pathlib import Path

import torch

from models.deepseek import DeepSeekConfig, DeepSeekLayer, load_weights


MODEL_DIR = Path("/data/xinference/cache/deepseek-v2-chat-pytorch-16b")


def test_deepseek_single_moe_layer_real_weights_forward() -> None:
    config = DeepSeekConfig.from_json(MODEL_DIR / "config.json")
    layer_idx = 1  # layer 0 is dense MLP, layer 1 starts using MoE
    layer = DeepSeekLayer(config, layer_idx=layer_idx).eval()

    loaded = load_weights(layer, MODEL_DIR, layer_idx=layer_idx)
    assert "self_attn.q_proj.weight" in loaded
    assert "mlp.gate.weight" in loaded
    assert "mlp.shared_experts.gate_proj.weight" in loaded
    assert "mlp.experts.0.gate_proj.weight" in loaded

    batch_size = 2
    seq_len = 5
    fake_token_ids = torch.randint(low=0, high=1024, size=(batch_size, seq_len))
    embed = torch.nn.Embedding(1024, config.hidden_size)
    hidden_states = embed(fake_token_ids).to(dtype=torch.float32)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)

    with torch.no_grad():
        out = layer(hidden_states, position_ids=position_ids)

    print("input shape:", tuple(hidden_states.shape))
    print("output shape:", tuple(out.shape))
    assert out.shape == hidden_states.shape


if __name__ == "__main__":
    test_deepseek_single_moe_layer_real_weights_forward()
