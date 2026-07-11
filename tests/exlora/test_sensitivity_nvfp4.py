import pytest
import torch

from shard.exlora.qat_map import QATMap, tensor_map_entry
from shard.exlora.sensitivity_nvfp4 import (
    apply_block_residual_sidecar,
    block_residual_bpe,
    block_scores,
    elementwise_weighted_relative_error,
    select_blocks_by_budget,
)


def test_block_scores_reduce_on_flattened_grid():
    x = torch.arange(1, 9, dtype=torch.float32).reshape(2, 4)
    assert torch.allclose(block_scores(x, block_size=4, reduce="mean"), torch.tensor([2.5, 6.5]))
    assert torch.allclose(block_scores(x, block_size=4, reduce="max"), torch.tensor([4.0, 8.0]))
    assert torch.allclose(block_scores(x, block_size=4, reduce="sum"), torch.tensor([10.0, 26.0]))


def test_select_blocks_by_budget_uses_extra_bpe_cap():
    scores = torch.tensor([0.1, 9.0, 3.0, 1.0])
    # 16 elems, block_size=4, residual_bits=8, extra_bpe=2 -> one block.
    mask = select_blocks_by_budget(scores, numel=16, extra_bpe=2.0, residual_bits=8, block_size=4)
    assert mask.tolist() == [False, True, False, False]
    assert block_residual_bpe(mask, numel=16, residual_bits=8, block_size=4) == pytest.approx(2.0)


def test_select_blocks_by_budget_can_charge_index_bits():
    scores = torch.tensor([0.1, 9.0, 3.0, 1.0])
    # Each selected block costs 4*8 residual bits + 32 index bits = 64 bits.
    mask = select_blocks_by_budget(
        scores,
        numel=16,
        extra_bpe=4.0,
        residual_bits=8,
        block_size=4,
        index_bits=32,
    )
    assert mask.tolist() == [False, True, False, False]
    assert block_residual_bpe(
        mask,
        numel=16,
        residual_bits=8,
        block_size=4,
        index_bits=32,
    ) == pytest.approx(4.0)


def test_apply_block_residual_sidecar_only_selected_blocks():
    ref = torch.arange(8, dtype=torch.float32)
    q = torch.zeros_like(ref)
    mask = torch.tensor([True, False])
    recon = apply_block_residual_sidecar(ref, q, mask, block_size=4, residual_dtype=torch.float32)
    assert torch.allclose(recon[:4], ref[:4])
    assert torch.allclose(recon[4:], torch.zeros(4))


def test_weighted_relative_error_emphasizes_sensitive_parameters():
    ref = torch.tensor([1.0, 1.0])
    approx = torch.tensor([0.0, 1.0])
    weights = torch.tensor([10.0, 1.0])
    assert elementwise_weighted_relative_error(ref, approx, weights) > 0.9


def test_qat_map_block_sensitivity_matches_block_count():
    a = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4)
    qmap = QATMap("b", "q", {"L.weight": tensor_map_entry(a, torch.ones_like(a))})
    scores = qmap.block_sensitivity("L.weight", block_size=4, reduce="mean")
    assert scores.shape == (4,)
    assert qmap.block_sensitivity("missing") is None
