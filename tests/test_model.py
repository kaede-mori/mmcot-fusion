import pytest
import torch
from transformers import T5Config

from mmcot_fusion.models.t5_multimodal import T5ForMultimodalGeneration

PATCH_SIZE = (10, 32)


def tiny_config():
    return T5Config(
        vocab_size=128,
        d_model=32,
        d_kv=8,
        d_ff=64,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        decoder_start_token_id=0,
    )


@pytest.mark.parametrize("fusion", ["sigmoid_1h", "tanh_mh", "sigmoid_mh"])
def test_forward_loss(fusion):
    model = T5ForMultimodalGeneration(tiny_config(), PATCH_SIZE, fusion=fusion)
    input_ids = torch.randint(0, 128, (2, 12))
    attention_mask = torch.ones_like(input_ids)
    image_ids = torch.randn(2, *PATCH_SIZE)
    labels = torch.randint(0, 128, (2, 6))
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_ids=image_ids,
        labels=labels,
    )
    assert out.loss.isfinite()


def test_generate_passes_image_ids():
    model = T5ForMultimodalGeneration(tiny_config(), PATCH_SIZE, fusion="tanh_mh")
    input_ids = torch.randint(0, 128, (2, 12))
    image_ids = torch.randn(2, *PATCH_SIZE)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        image_ids=image_ids,
        max_length=8,
    )
    assert out.shape[0] == 2


def test_zero_init_tanh_model_matches_text_only_path():
    """With the gate closed, fused hidden states equal the raw encoder states."""
    torch.manual_seed(0)
    model = T5ForMultimodalGeneration(tiny_config(), PATCH_SIZE, fusion="tanh_mh")
    model.eval()
    input_ids = torch.randint(0, 128, (1, 12))
    attention_mask = torch.ones_like(input_ids)
    image_ids = torch.randn(1, *PATCH_SIZE)
    with torch.no_grad():
        encoder_hidden = model.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        fused = model.fusion(encoder_hidden, image_ids)
    torch.testing.assert_close(fused, encoder_hidden)


def test_checkpoint_key_remap(tmp_path):
    """A state dict with original MM-CoT key names loads into the fusion submodule."""
    config = tiny_config()
    src = T5ForMultimodalGeneration(config, PATCH_SIZE, fusion="sigmoid_1h")
    state = src.state_dict()
    renamed = {}
    for key, value in state.items():
        renamed[key.replace("fusion.", "")] = value
    torch.save(renamed, tmp_path / "pytorch_model.bin")
    config.save_pretrained(tmp_path)

    loaded = T5ForMultimodalGeneration.from_mmcot_checkpoint(str(tmp_path), PATCH_SIZE)
    torch.testing.assert_close(
        loaded.fusion.gate_dense.weight, src.fusion.gate_dense.weight
    )
    torch.testing.assert_close(loaded.shared.weight, src.shared.weight)
