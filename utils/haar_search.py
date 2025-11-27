import torch
from utils.haar_transform import *
from utils.split_search import *
from utils.mask import *

@torch.no_grad()
def BiLLM_with_row_haar_global(
    unsalient_block, salient_block
):
    W_unsalient_haar = torch.zeros_like(unsalient_block)

    W_low = haar_wavelet_transform_row_low(unsalient_block)
    W_low_binary, _ = process_with_double_global_optimal_split(W_low)
    
    W_unsalient_block_low = unsalient_block - inverse_haar_wavelet_transform_row(
        W_low_binary, torch.zeros_like(W_low_binary)
    )

    W_high = haar_wavelet_transform_row_high(W_unsalient_block_low)
    W_high_binary, _ = process_with_double_global_optimal_split(W_high)
    W_unsalient_haar = inverse_haar_wavelet_transform_row(W_low_binary, W_high_binary)

    salient_diff = salient_block - W_unsalient_haar
    W_salient_haar = process_with_optimal_split_salient_global(salient_diff)
    W_billm_haar = W_unsalient_haar + W_salient_haar
    return W_billm_haar

@torch.no_grad()
def BiLLM_with_row_haar_row(unsalient_block, salient_block):
    W_unsalient_haar = torch.zeros_like(unsalient_block)

    W_low = haar_wavelet_transform_row_low(unsalient_block)
    W_low_binary, _ = process_with_double_row_optimal_split(W_low)
    
    W_unsalient_block_low = unsalient_block - inverse_haar_wavelet_transform_row(
        W_low_binary, torch.zeros_like(W_low_binary)
    )

    W_high = haar_wavelet_transform_row_high(W_unsalient_block_low)
    W_high_binary, _ = process_with_double_row_optimal_split(W_high)
    W_unsalient_haar = inverse_haar_wavelet_transform_row(W_low_binary, W_high_binary)

    salient_diff = salient_block - W_unsalient_haar
    W_salient_haar = process_with_optimal_split_salient(salient_diff)
    W_billm_haar = W_unsalient_haar + W_salient_haar
    return W_billm_haar

@torch.no_grad()
def BiLLM_with_row_haar_row_shared_mean(unsalient_block, salient_block):
    W_unsalient_haar = torch.zeros_like(unsalient_block)

    W_low = haar_wavelet_transform_row_low(unsalient_block)
    W_low_binary, _ = process_with_double_row_optimal_split_shared_mean(W_low)
    
    W_unsalient_block_low = unsalient_block - inverse_haar_wavelet_transform_row(
        W_low_binary, torch.zeros_like(W_low_binary)
    )

    W_high = haar_wavelet_transform_row_high(W_unsalient_block_low)
    W_high_binary, _ = process_with_double_row_optimal_split_shared_mean(W_high)
    W_unsalient_haar = inverse_haar_wavelet_transform_row(W_low_binary, W_high_binary)

    salient_diff = salient_block - W_unsalient_haar
    W_salient_haar = process_with_optimal_split_salient(salient_diff)
    W_billm_haar = W_unsalient_haar + W_salient_haar
    return W_billm_haar

@torch.no_grad()
def BiLLM_with_col_haar_global(
    origin_matrix, mask, unsalient_block, salient_block):
    W_unsalient_haar = torch.zeros_like(unsalient_block)
    W_unsalient_haar_restore = torch.zeros_like(origin_matrix)

    W_low = haar_wavelet_transform_col_low(unsalient_block)
    W_low_binary, _ = process_with_double_global_optimal_split(W_low)
    W_unsalient_block_low = unsalient_block - inverse_haar_wavelet_transform_col(
        W_low_binary, torch.zeros_like(W_low_binary)
    )

    W_high = haar_wavelet_transform_col_high(W_unsalient_block_low)
    W_high_binary, _ = process_with_double_global_optimal_split(W_high)
    W_unsalient_haar = inverse_haar_wavelet_transform_col(W_low_binary, W_high_binary)

    W_unsalient_haar_restore = restore_matrix(W_unsalient_haar, mask.any(dim=0))
    salient_diff = salient_block - W_unsalient_haar_restore
    W_salient_haar = process_with_optimal_split_salient_global(salient_diff)
    W_billm_haar = W_unsalient_haar_restore + W_salient_haar

    return W_billm_haar

@torch.no_grad()
def BiLLM_with_col_haar_row(
    origin_matrix, mask, unsalient_block, salient_block
):
    W_unsalient_haar = torch.zeros_like(unsalient_block)
    W_unsalient_haar_restore = torch.zeros_like(origin_matrix)

    W_low = haar_wavelet_transform_col_low(unsalient_block)
    W_low_binary, _ = process_with_double_row_optimal_split(W_low)
    W_unsalient_block_low = unsalient_block - inverse_haar_wavelet_transform_col(
        W_low_binary, torch.zeros_like(W_low_binary)
    )

    W_high = haar_wavelet_transform_col_high(W_unsalient_block_low)
    W_high_binary, _ = process_with_double_row_optimal_split(W_high)
    W_unsalient_haar = inverse_haar_wavelet_transform_col(W_low_binary, W_high_binary)

    W_unsalient_haar_restore = restore_matrix(W_unsalient_haar, mask.any(dim=0))
    salient_diff = salient_block - W_unsalient_haar_restore
    W_salient_haar = process_with_optimal_split_salient(salient_diff)
    W_billm_haar = W_unsalient_haar_restore + W_salient_haar

    return W_billm_haar

@torch.no_grad()
def BiLLM_with_col_haar_row_shared_mean(
    origin_matrix, mask, unsalient_block, salient_block
):
    W_unsalient_haar = torch.zeros_like(unsalient_block)
    W_unsalient_haar_restore = torch.zeros_like(origin_matrix)

    W_low = haar_wavelet_transform_col_low(unsalient_block)
    W_low_binary, _ = process_with_double_row_optimal_split_shared_mean(W_low)
    W_unsalient_block_low = unsalient_block - inverse_haar_wavelet_transform_col(
        W_low_binary, torch.zeros_like(W_low_binary)
    )

    W_high = haar_wavelet_transform_col_high(W_unsalient_block_low)
    W_high_binary, _ = process_with_double_row_optimal_split_shared_mean(W_high)

    W_unsalient_haar = inverse_haar_wavelet_transform_col(W_low_binary, W_high_binary)

    W_unsalient_haar_restore = restore_matrix(W_unsalient_haar, mask.any(dim=0))
    salient_diff = salient_block - W_unsalient_haar_restore
    W_salient_haar = process_with_optimal_split_salient(salient_diff)
    W_billm_haar = W_unsalient_haar_restore + W_salient_haar

    return W_billm_haar

