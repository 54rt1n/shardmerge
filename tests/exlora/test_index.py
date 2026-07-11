# tests/exlora/test_index.py
from shard.exlora.index import FlatModelIndex


def test_flat_index_tolerates_gemma_naming():
    """The base index raises on non-Llama naming; FlatModelIndex must not."""
    index = FlatModelIndex(download_manager=None, cache_path=None)
    index.model_indexes["m"] = {
        "weight_map": {
            "model.language_model.layers.0.self_attn.q_proj.weight": "s0",
            "model.language_model.layers.0.layer_scalar": "s0",
            "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight": "s0",
            "model.embed_tokens.weight": "s0",
        }
    }
    order = index._get_ordered_weights("m")
    # All weights preserved, no ordering-mismatch ValueError.
    assert set(order) == set(index.model_indexes["m"]["weight_map"].keys())
    assert len(order) == 4
