import torch

"""
Generate the structural mask on the basis of the split border
"""

def generate_structural_nomask(origin_matrix, braq1_border):
    """
    Generates two masks based on the absolute values of the original matrix and a given threshold.
    """

    binary_group = torch.abs(origin_matrix)

    mask2 = binary_group >= braq1_border
    mask1 = binary_group < braq1_border

    return mask1, mask2


def remove_nan_columns(tensor):
    """
    Removes all columns from the given tensor where every element is NaN.
    """

    mask = torch.isnan(tensor).all(dim=0)
    cleaned_tensor = tensor[:, ~mask]
    return mask, cleaned_tensor


def restore_matrix(W1, mask):
    """
    Restores the original matrix by setting the masked positions to zero.
    """

    restored_matrix = torch.zeros(
        W1.size(0), mask.size(0), dtype=W1.dtype, device=W1.device
    )
    restored_matrix[:, ~mask] = W1
    return restored_matrix


def generate_structural_nomask_row_abs(matrix, split_values):
    """
    Generates two row-wise masks based on a split value: one for values with absolute magnitude less than the threshold, 
    and one for values greater than or equal to it.
    """
    _, num_cols = matrix.shape
    split_values_expanded = split_values.unsqueeze(1).expand(-1, num_cols)

    binary_group = torch.abs(matrix)
    mask1 = binary_group < split_values_expanded
    mask2 = binary_group >= split_values_expanded

    return mask1, mask2
