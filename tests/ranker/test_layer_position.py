import pytest

from shard.ranker import bucket_depth, infer_num_layers, layer_position, parse_layer_index


def test_parse_layer_index_common_names():
    assert parse_layer_index("model.layers.12.self_attn.q_proj.weight") == 12
    assert parse_layer_index("transformer.h.3.mlp.c_fc.weight") == 3
    assert parse_layer_index("model.blocks.7.attn.q_proj.weight") == 7
    assert parse_layer_index("model.embed_tokens.weight") is None


def test_infer_num_layers_from_names():
    names = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.4.mlp.down_proj.weight",
        "model.norm.weight",
    ]
    assert infer_num_layers(names) == 5
    assert infer_num_layers(["model.norm.weight"]) is None


@pytest.mark.parametrize(
    ("fraction", "bucket"),
    [
        (None, "unknown"),
        (0.0, "early"),
        (0.19, "early"),
        (0.20, "middle"),
        (0.74, "middle"),
        (0.75, "late"),
        (1.0, "late"),
    ],
)
def test_bucket_depth(fraction, bucket):
    assert bucket_depth(fraction, early_fraction=0.20, late_fraction=0.75) == bucket


def test_layer_position_uses_inferred_count():
    pos = layer_position("model.layers.3.mlp.up_proj.weight", num_layers=10)

    assert pos.layer_index == 3
    assert pos.num_layers == 10
    assert pos.depth_fraction == pytest.approx(3 / 9)
    assert pos.depth_bucket == "middle"
