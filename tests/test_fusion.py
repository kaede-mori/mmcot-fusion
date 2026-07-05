import torch

from mmcot_fusion.models.fusion import (
    FUSION_REGISTRY,
    SigmoidGatedFusion,
    TanhGatedMHAFusion,
    build_fusion,
)

HIDDEN, PATCH_NUM, PATCH_DIM = 64, 10, 32
BATCH, SEQ = 2, 7


def _inputs():
    torch.manual_seed(0)
    hidden_states = torch.randn(BATCH, SEQ, HIDDEN)
    image_features = torch.randn(BATCH, PATCH_NUM, PATCH_DIM)
    return hidden_states, image_features


def test_registry_builds_all():
    for name in FUSION_REGISTRY:
        module = build_fusion(name, HIDDEN, PATCH_DIM)
        hidden_states, image_features = _inputs()
        out = module(hidden_states, image_features)
        assert out.shape == hidden_states.shape


def test_tanh_fusion_is_identity_at_init():
    module = TanhGatedMHAFusion(HIDDEN, PATCH_DIM, num_heads=8)
    hidden_states, image_features = _inputs()
    out = module(hidden_states, image_features)
    torch.testing.assert_close(out, hidden_states)


def test_tanh_fusion_departs_from_identity_when_gate_opens():
    module = TanhGatedMHAFusion(HIDDEN, PATCH_DIM, num_heads=8)
    with torch.no_grad():
        module.attn_gate.fill_(1.0)
    hidden_states, image_features = _inputs()
    out = module(hidden_states, image_features)
    assert not torch.allclose(out, hidden_states)


def test_sigmoid_fusion_matches_original_equations():
    torch.manual_seed(1)
    module = SigmoidGatedFusion(HIDDEN, PATCH_DIM, num_heads=1)
    hidden_states, image_features = _inputs()

    image_embedding = module.image_dense(image_features)
    image_att, _ = module.mha_layer(hidden_states, image_embedding, image_embedding)
    gate = module.sigmoid(module.gate_dense(torch.cat([hidden_states, image_att], dim=-1)))
    expected = (1 - gate) * hidden_states + gate * image_att

    torch.testing.assert_close(module(hidden_states, image_features), expected)


def test_gate_statistics():
    hidden_states, image_features = _inputs()
    stats = SigmoidGatedFusion(HIDDEN, PATCH_DIM).gate_statistics(hidden_states, image_features)
    assert 0.0 <= stats["gate_mean"] <= 1.0
    stats = TanhGatedMHAFusion(HIDDEN, PATCH_DIM).gate_statistics(hidden_states, image_features)
    assert stats["tanh_gate"] == 0.0
