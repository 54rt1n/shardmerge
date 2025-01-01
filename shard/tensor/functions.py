# shard/tensor/functions.py
# Copyright (C) 2024 Martin Bukowski
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

import logging
import torch
from typing import Generator, Literal
from .util import cuda_memory_profiler

logger = logging.getLogger(__name__)

def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    """
    Spherical linear interpolation (SLERP) between two vectors.
    
    Parameters:
    - v0 (torch.Tensor): The first vector.
    - v1 (torch.Tensor): The second vector.
    - t (float): Interpolation parameter (0 <= t <= 1).
    
    Returns:
    - torch.Tensor: Interpolated vector.
    """
    dot = torch.sum(v0 * v1) / (v0.norm() * v1.norm())
    dot = torch.clamp(dot, -1.0, 1.0)  # Clamp to avoid numerical issues
    
    theta = torch.acos(dot) * t
    relative_vec = v1 - v0 * dot
    relative_vec = torch.nn.functional.normalize(relative_vec, dim=-1)
    
    return v0 * torch.cos(theta) + relative_vec * torch.sin(theta)

def fft_transform(tensor: torch.Tensor, device: str) -> torch.Tensor:
    """
    Perform FFT on the tensor, considering its dimensionality.
    
    Parameters:
    - tensor (torch.Tensor): Input tensor.
    
    Returns:
    - torch.Tensor: FFT of the input tensor.
    """
    if tensor.ndim == 1:
        return torch.fft.fft(tensor.to(device).to(torch.float32)).to("cpu")
    else:
        return torch.fft.fftn(tensor.to(device).to(torch.float32), dim=(-2, -1)).to("cpu")

def ifft_transform(tensor: torch.Tensor, device: str) -> torch.Tensor:
    """
    Perform inverse FFT on the tensor, considering its dimensionality.
    
    Parameters:
    - tensor (torch.Tensor): Input tensor.
    
    Returns:
    - torch.Tensor: Inverse FFT of the input tensor.
    """
    if tensor.ndim == 1:
        return torch.fft.ifft(tensor.to(device)).real.to("cpu")
    else:
        return torch.fft.ifftn(tensor.to(device), dim=(-2, -1)).real.to("cpu")

def normalize_tensor(tensor: torch.Tensor, device: str) -> tuple[torch.Tensor, float]:
    """
    Normalize a tensor by its norm.
   <|start_header_id|>assistant<|end_header_id|>\n\n{bot_name}:  
    Parameters:
    - tensor (torch.Tensor): Input tensor.
    
    Returns:
    - torch.Tensor: Normalized tensor.
    """
    norm_t = tensor.to(device=device).norm()
    norm = norm_t.item()
    del norm_t
    return tensor / norm if norm != 0 else tensor, norm

def interpolate_fft_components(v0_fft: torch.Tensor, v1_fft: torch.Tensor, t: float, device: str, t_sum: float = 1.0, cutoff_pct: float = 0.0, cull_pct: float = 0.0, interp_imag : bool = True) -> torch.Tensor:
    """
    Interpolate the real and imaginary parts of FFT components using SLERP.
    
    Parameters:
    - v0_fft (torch.Tensor): FFT of the first tensor.
    - v1_fft (torch.Tensor): FFT of the second tensor.
    - t (float): Interpolation parameter (0 <= t <= 1).
    - t_sum (float): Sum of the interpolation parameters.
    - cutoff_pct (float): Floor percentage for doing task addition instead of slerp.
    - cull_pct (float): Floor percentage of the FFT components to cull.
    - interp_imag (bool): Whether to interpolate the imaginary parts of the FFT components.
    
    Returns:
    - torch.Tensor: Interpolated FFT components.
    """
    result_fft = torch.zeros_like(v0_fft, device=device)

    real_v0 = v0_fft.real.to(device)
    real_v1 = v1_fft.real.to(device)
    abs_real_v0 = real_v0.abs()
    abs_real_v1 = real_v1.abs()

    if cutoff_pct > 0:
        all_real, rest = torch.sort(torch.cat([abs_real_v0, abs_real_v1]).ravel(), descending=False)
        pct_idx = int(len(all_real) * cutoff_pct)
        if pct_idx >= len(all_real):
            cutoff_threshold = all_real[-1].item()
        else:
            cutoff_threshold = all_real[pct_idx].item()
        del all_real, rest
    else:
        cutoff_threshold = 0

    sign_mask = (real_v0.sign() == real_v1.sign()) 
    small_values_v0 = (abs_real_v1 < cutoff_threshold)
    small_values_v1 = (abs_real_v1 < cutoff_threshold)
    slerp_mask = sign_mask & ~small_values_v0 & ~small_values_v1
    sum_mask = sign_mask & ~slerp_mask
    rest_mask = ~slerp_mask & ~sum_mask
    del small_values_v0, small_values_v1, sign_mask
    larger_values_mask = abs_real_v0 > abs_real_v1
    
    # Interpolate real parts using SLERP
    result_fft.real[slerp_mask] = slerp( real_v0[slerp_mask],  real_v1[slerp_mask], t)
    result_fft.real[sum_mask] = real_v0[sum_mask] + t_sum * real_v1[sum_mask]
    result_fft.real[rest_mask] = torch.where(larger_values_mask[rest_mask], real_v0[rest_mask], real_v1[rest_mask])
    
    if cull_pct > 0:
        all_real, rest = torch.sort(result_fft.real.abs().ravel(), descending=False)
        cull_idx = int(len(all_real) * cull_pct)
        cull_threshold = all_real[cull_idx].item()
        # Check to make sure that the cull threshold doesn't take more than it should
        if (all_real < cull_threshold).sum() > (len(all_real) * (cull_pct * 2)):
            logger.info(f"Warning: Cull threshold overflow {cull_threshold} {cull_idx} {len(all_real)} {(all_real < cull_threshold).sum()} {len(all_real) * (cull_pct + 0.01)}")
        else:
            logger.debug(f"Cull {cull_threshold} {cull_idx} {len(all_real)} {(all_real < cull_threshold).sum()} {len(all_real) * (cull_pct + 0.01)}")
            result_fft.real[torch.abs(result_fft.real) < cull_threshold] = 0
        del all_real, rest

    del real_v0, real_v1, abs_real_v0, abs_real_v1, larger_values_mask, slerp_mask, sum_mask, rest_mask

    if interp_imag:
        i0_fft = fft_transform(v0_fft.imag, device=device)
        i1_fft = fft_transform(v1_fft.imag, device=device)
        # The zero cutoff is more about memory issue then for a good reason
        i0_fft = interpolate_fft_components(i0_fft, i1_fft, t=t, cutoff_pct=0, cull_pct=0, interp_imag=False, device=device)
        del i1_fft
        result_fft.imag = ifft_transform(i0_fft, device=device)
    else:
        result_fft.imag = v0_fft.imag

    return result_fft

def merge_tensors_fft2_slerp(v0: torch.Tensor, v1: torch.Tensor, t: float, device: str, b: float = .1, t_sum : float = 1.0,  cutoff_pct: float = 0.0, cull_pct: float = 0.0) -> tuple[torch.Tensor, float, float]:
    """
    Merges two tensors using 2D Fourier transform interpolation with SLERP for both
    the real and imaginary parts.
    
    Parameters:
    - v0 (torch.Tensor): The first input tensor.
    - v1 (torch.Tensor): The second input tensor.
    - t (float): Interpolation parameter (0 <= t <= 1).
    - b (float): Threshold for the ratio between the norms of the input tensors.
    - t_sum (float): Interpolation parameter for the sum of the two tensors.
    
    Returns:
    - Tuple[torch.Tensor, float, float]: The interpolated tensor and the norms of the input tensors.
    """

    # Normalize tensors
    v0, norm_v0 = normalize_tensor(v0, device=device)
    v1, norm_v1 = normalize_tensor(v1, device=device)
    
    if norm_v1 < .0001:
        return v0, norm_v0, norm_v1
    
    if norm_v0 < .0001:
        # I really don't know what to do here
        logger.info(f"Warning: Small norm v0 ({norm_v0})")
        return v0, norm_v0, norm_v1
    
    # Compute FFT of the input tensors
    fft_v0 = fft_transform(v0, device=device)
    fft_v1 = fft_transform(v1, device=device)

    ratio = norm_v1 / (norm_v0 + 1e-10)

    # If we have really small values, they are probably noise, but might be important
    if ratio < b:
        logger.info(f"Small norm v1 ({norm_v1})")
        # We should just add the second tensor to the first one
        result_fft = fft_v0 + fft_v1 * t
    else:
        # Interpolate FFT components
        result_fft = interpolate_fft_components(fft_v0, fft_v1, t=t, t_sum=t_sum, cutoff_pct=cutoff_pct, cull_pct=cull_pct, device=device)
        
    # Perform the inverse FFT to get back to the spatial domain
    merged_tensor = ifft_transform(result_fft, device=device)

    # check for nan/inf and print a report
    if torch.any(torch.isnan(merged_tensor)):
        merged_tensor = torch.where(torch.isnan(merged_tensor), torch.zeros_like(merged_tensor), merged_tensor)
        logger.info(f"Warning: NaN in ifft output: {torch.isnan(merged_tensor).sum()}")
    
    if torch.any(torch.isinf(merged_tensor)):
        logger.info(f"Warning: Inf in ifft output: {torch.isinf(merged_tensor).sum()}")
        raise ValueError("Inf in ifft output")

    del fft_v0, fft_v1, result_fft

    return merged_tensor, norm_v0, norm_v1


def task_arithmetic_fft2(v0: torch.Tensor, v1: torch.Tensor, t: float, device: str, agreement : bool = True) -> torch.Tensor:
    """
    Perform arithmetic task using FFT2 with sign agreement and summing matching signs.
    
    Parameters:
    - v0 (torch.Tensor): The first input tensor.
    - v1 (torch.Tensor): The second input tensor.
    - t (float): Interpolation parameter (0 <= t <= 1).
    - agreement (bool): Whether to apply sign agreement.
    
    Returns:
    - torch.Tensor: The resulting tensor after applying the arithmetic task.
    """

    # Compute FFT of the input tensors
    logger.debug("FFT v0")
    fft_v0 = fft_transform(v0, device=device)
    logger.debug("FFT v1")
    fft_v1 = fft_transform(v1, device=device)

    # Perform arithmetic task on FFT components
    logger.debug("arithmetic_fft_components")
    result_fft = arithmetic_fft_components(fft_v0, fft_v1, t=t, agreement=agreement, device=device)
    del fft_v0, fft_v1

    # Perform the inverse FFT to get back to the spatial domain
    logger.debug("ifft_transform")
    result = ifft_transform(result_fft, device=device)
    del result_fft

    return result

def arithmetic_fft_components(v0_fft: torch.Tensor, v1_fft: torch.Tensor, t: float, agreement: bool, device: str, do_imag : bool = True) -> torch.Tensor:
    """
    Perform arithmetic task on the real and imaginary parts of FFT components.
    
    Parameters:
    - v0_fft (torch.Tensor): FFT of the first tensor.
    - v1_fft (torch.Tensor): FFT of the second tensor.
    - t (float): Interpolation parameter (0 <= t <= 1).
    - agreement (bool): Whether to apply sign agreement.
    - do_imag (bool): Whether to perform the arithmetic task on the imaginary part.
    
    Returns:
    - torch.Tensor: Resulting FFT components after applying the arithmetic task.
    """
    with cuda_memory_profiler(display=False):
        result_fft = torch.zeros_like(v0_fft, device="cpu")

        logger.debug("sign_mask")
        if agreement:
            sign_mask = v0_fft.real.sign() == v1_fft.real.sign()
        else:
            sign_mask = torch.ones_like(v0_fft.real, dtype=torch.bool).bool()

        # Sum real parts where signs agree, else take the larger magnitude
        result_fft.real[sign_mask] = v0_fft.real[sign_mask] + t * v1_fft.real[sign_mask]
        logger.debug("summed")
        larger_values_mask = (v0_fft.real.abs() > v0_fft.real.abs())

        result_fft.real[~sign_mask] = torch.where(larger_values_mask[~sign_mask], v0_fft.real[~sign_mask], v1_fft.real[~sign_mask]).to("cpu")
        logger.debug("masked")
        del sign_mask, larger_values_mask

    logger.debug("updated")

    if do_imag:
        with cuda_memory_profiler(display=False):
            # Take the second Fourier transform of the imaginary parts
            logger.debug("imag")
            i0_fft = fft_transform(v0_fft.imag, device=device)
            i1_fft = fft_transform(v1_fft.imag, device=device)
            i0_fft = arithmetic_fft_components(i0_fft, i1_fft, t=t, agreement=agreement, do_imag=False, device=device)
            del i1_fft
            result_fft.imag = ifft_transform(i0_fft, device=device)
    else:
        result_fft.imag = v0_fft.imag

    return result_fft

def correlate_pairs(tensors : torch.Tensor, work_device : str, store_device : str) -> torch.Tensor:
    n = tensors.shape[0]
    matrix = torch.zeros(n, n).to(store_device)
    for i in range(n):
        a = tensors[i].to(work_device)
        for j in range(i + 1, n):
            b = tensors[j].to(work_device)
            matrix[i, j] = matrix[j, i] = torch.nn.functional.cosine_similarity(a, b, dim=0).nan_to_num(0).mean().item()
            b = b.to(store_device)
        a = a.to(store_device)
    return matrix

def correlated_pairs(correlation_matrix: torch.Tensor, way: Literal['least', 'most'] = 'least') -> Generator[tuple[int, int, float], None, None]:
    """
    Generates tuples of indices and their corresponding least correlation coefficient
    from a given correlation matrix, ensuring that once an index is used, it is no longer
    considered in future tuples.

    Args:
        correlation_matrix (torch.Tensor): A 2D square tensor representing the correlation matrix.

    Yields:
        Tuple[int, int, float]: A tuple containing the x-index, y-index, and the correlation coefficient of the least correlated pairs in the matrix.
    """
    n = correlation_matrix.size(0)
    # Create a mask to exclude diagonal and already processed elements
    available_indices = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    items = list(range(n))
    
    while torch.any(available_indices):
        # Apply mask to get relevant correlations
        valid_correlation = torch.where(available_indices, correlation_matrix, torch.tensor(float('inf')))
        
        if way == 'least':
            # Find the minimum non-zero absolute correlation
            match_val = torch.min(torch.abs(valid_correlation[valid_correlation != float('inf')]))
        elif way == 'most':
            # Find the maximum absolute correlation
            match_val = torch.max(torch.abs(valid_correlation[valid_correlation != float('inf')]))
        else:
            raise ValueError("Invalid way. Choose 'least' or 'most'.")
            
        # Locate the indices with the minimum correlation
        match_indices = torch.nonzero(torch.abs(valid_correlation) == match_val, as_tuple=True)
        if len(match_indices[0]) == 0:
            break
        
        # Yield the first index pair (greedy approach) along with the correlation coefficient
        x, y = match_indices[0][0].item(), match_indices[1][0].item()
        coefficient = correlation_matrix[x, y].item()  # Extract the actual correlation value
        yield (x, y, coefficient)
        
        # Mask out the entire row and column for both indices
        available_indices[x, :] = False
        available_indices[:, x] = False
        available_indices[y, :] = False
        available_indices[:, y] = False
        items.remove(x)
        items.remove(y)
    if len(items) > 0:
        for i in items:
            yield (i, -1, correlation_matrix[i, i].item())
