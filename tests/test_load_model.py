"""load_model gives the same model as from_pretrained, on CPU, for a tiny random Qwen2 split
over many files. Needs torch and transformers (the CI 'loader' job installs them)."""

import glob
import os

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from nrp_kit.load_model import load_model  # noqa: E402


@pytest.mark.parametrize("tied", [False, True])
def test_matches_from_pretrained(tmp_path, tied):
    cfg = transformers.Qwen2Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                                   tie_word_embeddings=tied, torch_dtype="bfloat16")
    torch.manual_seed(0)
    ref = transformers.AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16).eval()
    ref.save_pretrained(tmp_path, max_shard_size="40KB")
    shards = sorted(glob.glob(os.path.join(tmp_path, "*.safetensors")))
    assert len(shards) > 4
    std = transformers.AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.bfloat16).eval()
    ours = load_model(str(tmp_path), shards, "cpu")
    if tied:
        assert ours.lm_head.weight is ours.model.embed_tokens.weight
    x = torch.randint(0, 512, (2, 17))
    with torch.no_grad():
        assert torch.equal(std(x).logits, ours(x).logits)


def test_missing_file_fails(tmp_path):
    cfg = transformers.Qwen2Config(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=False)
    transformers.AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16).save_pretrained(
        tmp_path, max_shard_size="20KB")
    shards = sorted(glob.glob(os.path.join(tmp_path, "*.safetensors")))
    with pytest.raises(ValueError, match="missing"):
        load_model(str(tmp_path), shards[:-1], "cpu")
