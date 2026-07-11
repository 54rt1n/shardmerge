# tests/exlora/test_nvfp4_kernel.py
import pytest
import torch

from shard.exlora.nvfp4 import nvfp4_quantize, nvfp4_quantize_dequantize
from shard.exlora.nvfp4_kernel import (
    NVFP4PackedTensor,
    is_triton_nvfp4_available,
    nvfp4_quantize_dequantize_triton,
    nvfp4_quantize_triton,
    pack_nvfp4_codes,
    pack_reference_nvfp4,
    unpack_nvfp4_codes,
)


class TestNVFP4Packing:
    def test_pack_unpack_roundtrip(self):
        codes = torch.arange(32, dtype=torch.int8).reshape(2, 16) & 0x0F
        packed = pack_nvfp4_codes(codes)
        assert packed.dtype == torch.uint8
        assert packed.shape == (2, 8)
        assert torch.equal(unpack_nvfp4_codes(packed), codes)

    def test_unpack_rejects_wrong_block_size(self):
        packed = torch.zeros(2, 8, dtype=torch.uint8)
        with pytest.raises(ValueError, match="block_size"):
            unpack_nvfp4_codes(packed, block_size=8)

    def test_pack_rejects_odd_last_dimension(self):
        with pytest.raises(ValueError, match="even"):
            pack_nvfp4_codes(torch.zeros(2, 15, dtype=torch.int8))

    def test_pack_rejects_non_integer_codes(self):
        with pytest.raises(TypeError, match="integer"):
            pack_nvfp4_codes(torch.zeros(2, 16, dtype=torch.float32))

    def test_pack_rejects_out_of_range_codes(self):
        codes = torch.zeros(2, 16, dtype=torch.int8)
        codes[0, 0] = 16
        with pytest.raises(ValueError, match=r"\[0, 15\]"):
            pack_nvfp4_codes(codes)


class TestPackedNVFP4Tensor:
    def test_pack_reference_dequant_matches_reference(self):
        torch.manual_seed(0)
        x = torch.randn(5, 17)
        ref = nvfp4_quantize(x)
        packed = pack_reference_nvfp4(ref)
        assert isinstance(packed, NVFP4PackedTensor)
        assert torch.equal(unpack_nvfp4_codes(packed.packed_codes), ref.codes)
        assert torch.allclose(packed.dequantize(), ref.dequantize())

    def test_format_bits_count_padding(self):
        x = torch.ones(1)
        packed = pack_reference_nvfp4(nvfp4_quantize(x))
        # One padded 16-value block: 8 packed-code bytes + 1 fp8 scale + 4 global bytes.
        assert packed.bits_per_element == pytest.approx(104.0)

    def test_tensor_bits_tracks_scale_dtype(self):
        x = torch.randn(32)
        packed = pack_reference_nvfp4(nvfp4_quantize(x))
        assert packed.tensor_bits_per_element >= packed.format_bits_per_element

    def test_dequant_rejects_mismatched_scale_count(self):
        packed = NVFP4PackedTensor(
            packed_codes=torch.zeros(2, 8, dtype=torch.uint8),
            block_scale=torch.ones(1),
            global_scale=1.0,
            shape=torch.Size([32]),
        )
        with pytest.raises(ValueError, match="nblocks"):
            packed.dequantize()

    def test_dequant_rejects_non_uint8_codes(self):
        packed = NVFP4PackedTensor(
            packed_codes=torch.zeros(2, 8, dtype=torch.int8),
            block_scale=torch.ones(2),
            global_scale=1.0,
            shape=torch.Size([32]),
        )
        with pytest.raises(TypeError, match="uint8"):
            packed.dequantize()

    def test_dequant_rejects_non_1d_scales(self):
        packed = NVFP4PackedTensor(
            packed_codes=torch.zeros(2, 8, dtype=torch.uint8),
            block_scale=torch.ones(2, 1),
            global_scale=1.0,
            shape=torch.Size([32]),
        )
        with pytest.raises(ValueError, match="1D"):
            packed.dequantize()


@pytest.mark.skipif(not is_triton_nvfp4_available(), reason="CUDA Triton NVFP4 path unavailable")
class TestTritonNVFP4:
    def test_triton_roundtrip_shape_and_dtype(self):
        x = torch.randn(33, device="cuda", dtype=torch.float16)
        out = nvfp4_quantize_dequantize_triton(x)
        assert out.shape == x.shape
        assert out.dtype == x.dtype

    def test_triton_packed_bits_about_four_and_half(self):
        x = torch.randn(32, 256, device="cuda")
        q = nvfp4_quantize_triton(x)
        assert q.packed_codes.dtype == torch.uint8
        assert 4.4 < q.bits_per_element < 4.7

    def test_triton_error_reasonable(self):
        torch.manual_seed(0)
        x = torch.randn(64, 256, device="cuda")
        out = nvfp4_quantize_dequantize_triton(x)
        rel = (out - x).norm() / x.norm()
        assert rel.item() < 0.16

    def test_triton_close_to_reference_codec(self):
        torch.manual_seed(1)
        x = torch.randn(4, 128, device="cuda")
        triton_out = nvfp4_quantize_dequantize_triton(x)
        ref_out = nvfp4_quantize_dequantize(x)
        rel = (triton_out - ref_out).norm() / ref_out.norm().clamp_min(1e-12)
        assert rel.item() < 0.08

    def test_triton_packed_payload_matches_reference_codec(self):
        torch.manual_seed(2)
        x = torch.randn(513, device="cuda")
        triton_q = nvfp4_quantize_triton(x)
        ref_q = pack_reference_nvfp4(nvfp4_quantize(x))
        assert torch.equal(triton_q.packed_codes.cpu(), ref_q.packed_codes.cpu())
        assert torch.equal(triton_q.block_scale.cpu(), ref_q.block_scale.cpu())

    def test_triton_packed_payload_matches_reference_codec_many_blocks(self):
        torch.manual_seed(3)
        x = torch.randn(1 << 20, device="cuda")
        triton_q = nvfp4_quantize_triton(x)
        ref_q = pack_reference_nvfp4(nvfp4_quantize(x))
        assert torch.equal(triton_q.packed_codes.cpu(), ref_q.packed_codes.cpu())
        assert torch.equal(triton_q.block_scale.cpu(), ref_q.block_scale.cpu())

    def test_triton_empty_tensor_roundtrip(self):
        x = torch.empty(0, device="cuda")
        q = nvfp4_quantize_triton(x)
        out = q.dequantize(dtype=x.dtype)
        assert out.shape == x.shape
        assert out.dtype == x.dtype
