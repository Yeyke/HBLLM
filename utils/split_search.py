import torch
import numpy as np
from utils.mask import *
from utils.haar_transform import *
from utils.binary import high_order_residual_rms,high_order_residual_rms_shared_mean


def error_computing(origin_matrix, quantized_matrix):
    mse = torch.mean((origin_matrix - quantized_matrix) ** 2)
    return mse

def error_computing_row(origin_matrix, quantized_matrix):
    l2 = torch.sum((origin_matrix - quantized_matrix) ** 2, dim=1)
    return l2

@torch.no_grad()
def fill_significant_columns_with_avg(matrix_block, mask1, mask2, window=7):
    filled_matrix_block = matrix_block.clone()
    n_cols = matrix_block.shape[1]
    salient_matrix_block = torch.full_like(matrix_block, float("nan"))

    for col_idx in range(n_cols):
        if mask1[:, col_idx].any():
            start = max(0, col_idx - window // 2)
            end = min(n_cols, col_idx + window // 2 + 1)
            non_significant_cols = mask2[:, start:end].any(dim=0)

            if non_significant_cols.any():
                non_significant_indices = non_significant_cols.nonzero(
                    as_tuple=False
                ).squeeze()
                if non_significant_indices.numel() == 1:
                    fill_values = matrix_block[
                        :, start + non_significant_indices.item()
                    ]
                else:
                    fill_values = matrix_block[:, start + non_significant_indices].mean(
                        dim=1
                    )
                filled_matrix_block[:, col_idx] = fill_values
            else:
                if col_idx == 0:
                    non_significant_col = (
                        (mask2[:, col_idx + 1 :].any(dim=0))
                        .nonzero(as_tuple=False)[0]
                        .item()
                        + col_idx
                        + 1
                    )
                    filled_matrix_block[:, col_idx] = matrix_block[
                        :, non_significant_col
                    ].clone()
                else:
                    left_cols = mask2[:, :col_idx].any(dim=0)
                    if left_cols.any():
                        non_significant_col = left_cols.nonzero(as_tuple=False)[
                            -1
                        ].item()
                        filled_matrix_block[:, col_idx] = matrix_block[
                            :, non_significant_col
                        ].clone()
                    else:
                        non_significant_col = (
                            (mask2[:, col_idx + 1 :].any(dim=0))
                            .nonzero(as_tuple=False)[0]
                            .item()
                            + col_idx
                            + 1
                        )
                        filled_matrix_block[:, col_idx] = matrix_block[
                            :, non_significant_col
                        ].clone()

            salient_matrix_block[:, col_idx] = matrix_block[:, col_idx]

    return filled_matrix_block, salient_matrix_block

def separate_columns_withnan(matrix_block, mask1, mask2):
    """
    Given two complementary column masks, splits the input matrix into:

    1. filled_matrix_block: contains columns where mask2 is True.
    2. salient_matrix_block: same shape as the input matrix, where only entries with mask1 are preserved; others are set to NaN.
    """

    filled_matrix_block = matrix_block[:, mask2.any(dim=0)]
    salient_matrix_block = torch.full_like(matrix_block, float("nan"))
    salient_matrix_block[mask1] = matrix_block[mask1]

    return filled_matrix_block, salient_matrix_block

@torch.no_grad()
def process_with_double_global_optimal_split(matrix):
    """
    Selects the optimal split_value based on minimal quantization error for first-order binarization.

    Args:
    - global: use global partitioning criteria.

    Returns:
    - torch.Tensor: The optimal combination of quantized matrices.
    - optimal_split_value (float): The best split threshold.
    """
    flat_abs_tensor = torch.abs(matrix).view(-1)
    percentiles = torch.linspace(0.10, 0.90, 81).to(matrix.device)
    percentile_values = torch.tensor(
        np.quantile(
            flat_abs_tensor.detach().cpu().numpy(),
            q=percentiles.cpu().numpy(),
            axis=None,
            keepdims=False,
        )
    ).to(matrix.device)

    minimal_error = float("inf")
    optimal_split_value = percentile_values[0]
    optimal_binary = torch.zeros_like(matrix)

    for split_value in percentile_values:
        mask1, mask2 = generate_structural_nomask(matrix, split_value)
        group1 = high_order_residual_rms(matrix, mask1, order=1)
        group2 = high_order_residual_rms(matrix, mask2, order=1)
        W_binary = group1 + group2

        quantize_error = error_computing(matrix, W_binary)

        if quantize_error < minimal_error:
            minimal_error = quantize_error
            optimal_split_value = split_value
            optimal_binary = W_binary

    return optimal_binary, optimal_split_value

@torch.no_grad()
def process_with_double_row_optimal_split(matrix):
    """
    Selects the optimal split_value based on minimal quantization error for first-order binarization.

    Args:
    - row: use row-wise partitioning criteria.

    Returns:
    - torch.Tensor: The optimal combination of quantized matrices.
    - optimal_split_value (float): The best split threshold.
    """
    num_rows, _ = matrix.shape
    optimal_binary = torch.zeros_like(matrix)
    optimal_split_values = torch.zeros(num_rows, device=matrix.device)
    minimal_errors = torch.full((num_rows,), float("inf"), device=matrix.device)

    abs_tensor = torch.abs(matrix)
    percentiles = torch.linspace(0.10, 0.90, 41).to(matrix.device)
    row_split_values = torch.quantile(abs_tensor, percentiles, dim=1)

    for split_value in row_split_values:
        mask1, mask2 = generate_structural_nomask_row_abs(matrix, split_value)

        group1 = high_order_residual_rms(matrix, mask1, order=1)
        group2 = high_order_residual_rms(matrix, mask2, order=1)
        W_binary = group1 + group2
        quantize_error_row = error_computing_row(matrix, W_binary)

        mask_update = quantize_error_row < minimal_errors
        minimal_errors = torch.where(mask_update, quantize_error_row, minimal_errors)
        optimal_binary = torch.where(mask_update.view(-1, 1), W_binary, optimal_binary)
        optimal_split_values = torch.where(
            mask_update, split_value.view(1, -1), optimal_split_values
        )
    return optimal_binary, optimal_split_values

@torch.no_grad()
def process_with_double_row_optimal_split_shared_mean(matrix):
    """
    Selects the optimal split_value based on minimal quantization error for first-order binarization.

    Args:
    - row: use row-wise partitioning criteria.
    - shared mean: each row different group share mean 

    Returns:
    - torch.Tensor: The optimal combination of quantized matrices.
    - optimal_split_value (float): The best split threshold.
    """
    num_rows, _ = matrix.shape
    optimal_binary = torch.zeros_like(matrix)
    optimal_mean = torch.zeros(num_rows, device=matrix.device)
    optimal_split_values = torch.zeros(num_rows, device=matrix.device)
    minimal_errors = torch.full((num_rows,), float("inf"), device=matrix.device)
    optimal_mean = torch.mean(matrix, dim=1)

    abs_tensor = torch.abs(matrix)
    percentiles = torch.linspace(0.10, 0.90, 41).to(matrix.device)
    row_split_values = torch.quantile(abs_tensor, percentiles, dim=1)

    for split_value in row_split_values:
        mask1, mask2 = generate_structural_nomask_row_abs(matrix, split_value)

        group1 = high_order_residual_rms_shared_mean(
            matrix, mask1, optimal_mean
        )
        group2 = high_order_residual_rms_shared_mean(
            matrix, mask2, optimal_mean
        )
        W_binary = group1 + group2

        quantize_error_row = error_computing_row(matrix, W_binary)

        mask_update = quantize_error_row < minimal_errors
        minimal_errors = torch.where(mask_update, quantize_error_row, minimal_errors)
        optimal_binary = torch.where(mask_update.view(-1, 1), W_binary, optimal_binary)
        optimal_split_values = torch.where(
            mask_update, split_value.view(1, -1), optimal_split_values
        )

    return optimal_binary, optimal_split_values

@torch.no_grad()
def process_with_optimal_split_salient(matrix):
    mask, cleaned_matrix = remove_nan_columns(matrix)
    W_low = haar_wavelet_transform_col_low(cleaned_matrix)
    W_low_binary, _ = process_with_double_row_optimal_split(W_low)

    W_diff = cleaned_matrix - inverse_haar_wavelet_transform_col(
        W_low_binary, torch.zeros_like(W_low_binary)
    )

    W_high = haar_wavelet_transform_col_high(W_diff)
    W_high_binary, _ = process_with_double_row_optimal_split(W_high)
    W_haar = inverse_haar_wavelet_transform_col(W_low_binary, W_high_binary)
    W_haar_binary = restore_matrix(W_haar, mask)

    return W_haar_binary

@torch.no_grad()
def process_with_optimal_split_salient_global(matrix):
    mask, cleaned_matrix = remove_nan_columns(matrix)
    W_low = haar_wavelet_transform_col_low(cleaned_matrix)

    W_low_binary, _= process_with_double_global_optimal_split(W_low)

    W_diff = cleaned_matrix - inverse_haar_wavelet_transform_col(
        W_low_binary, torch.zeros_like(W_low_binary)
    )

    W_high = haar_wavelet_transform_col_high(W_diff)
    W_high_binary, _ = process_with_double_global_optimal_split(W_high)
    W_haar = inverse_haar_wavelet_transform_col(W_low_binary, W_high_binary)
    W_haar_binary = restore_matrix(W_haar, mask)

    return W_haar_binary