from re import L
import torch
from utils.binary import high_order_residual


def error_computing(origin_matrix, quantized_matrix):
    mse = torch.mean((origin_matrix - quantized_matrix) ** 2)
    return mse


def structural_searching(origin_matrix, up_lim=30, metric="l2"):
    minimal_value_0 = float("inf")

    if metric == "l1":
        true_counts = origin_matrix.abs().sum(dim=0)
    elif metric == "l2":
        true_counts = torch.norm(origin_matrix, p=2, dim=0)
    else:
        raise NotImplementedError
    
    error = []
    lines = []
    _, top_braq_2_columns = torch.topk(true_counts, up_lim)
    for i in range(1, up_lim):
        mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(
            origin_matrix.device
        )
        mask3[:, top_braq_2_columns[:i]] = True
        group3 = high_order_residual(origin_matrix, mask3, order=2)
        group4 = high_order_residual(origin_matrix, ~mask3, order=2)
        quantize_error_0 = error_computing(origin_matrix, group4 + group3)
        error.append(quantize_error_0.item())
        lines.append(i)
        if quantize_error_0 < minimal_value_0:
            minimal_value_0 = quantize_error_0
            optimal_split_0 = i
    _, top_braq_2_columns = torch.topk(true_counts, optimal_split_0)
    mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(
        origin_matrix.device
    )
    mask3[:, top_braq_2_columns] = True

    return mask3
