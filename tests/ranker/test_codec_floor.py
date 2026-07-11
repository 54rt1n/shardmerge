import torch

from shard.ranker import CodecFloorConfig, CodecFloorModel


def test_codec_floor_detects_background_and_super_floor_blocks():
    scores = torch.tensor([1.00, 1.02, 0.99, 1.01, 1.03, 0.98, 1.00, 1.01, 3.5, 4.0])

    model = CodecFloorModel.fit(
        scores,
        CodecFloorConfig(floor_tolerance=3.0, effect_threshold=5.0, min_floor_blocks=4),
    )

    assert model.floor_blocks >= 8
    assert model.super_floor_blocks == 2
    assert model.super_floor_mask[-2:].all()
    assert model.effect_scores[-1] > model.effect_scores[0]


def test_codec_floor_handles_empty_scores():
    model = CodecFloorModel.fit(torch.empty(0))

    assert model.floor_level == 0.0
    assert model.floor_blocks == 0
    assert model.super_floor_blocks == 0
