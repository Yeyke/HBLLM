import torch

@torch.no_grad()
def haar_wavelet_transform_row_low(W):
    """
    Applies a row-wise Haar wavelet transform to the input matrix and returns only the low-frequency component (W_low).
    """
    _, n_cols = W.shape
    if n_cols % 2 != 0:
        raise ValueError("The number of columns must be even for Haar transform.")

    even_cols = W[:, ::2]
    odd_cols = W[:, 1::2]

    norm_factor = 0.70710677
    W_low = torch.mul(even_cols + odd_cols, norm_factor)

    return W_low


@torch.no_grad()
def haar_wavelet_transform_row_high(W):
    """
    Applies a row-wise Haar wavelet transform to the input matrix and returns only the high-frequency component (W_high).
    """
    _, n_cols = W.shape
    if n_cols % 2 != 0:
        raise ValueError("The number of columns must be even for Haar transform.")

    even_cols = W[:, ::2]
    odd_cols = W[:, 1::2]

    norm_factor = 0.70710677
    W_high = torch.mul(even_cols - odd_cols, norm_factor)

    return W_high


@torch.no_grad()
def haar_wavelet_transform_row(W):
    """
    row-wise Haar wavelet transform
    """
    _, n_cols = W.shape
    if n_cols % 2 != 0:
        raise ValueError("The number of columns must be even for Haar transform.")

    even_cols = W[:, ::2]
    odd_cols = W[:, 1::2]

    norm_factor = 0.70710677
    W_low = torch.mul(even_cols + odd_cols, norm_factor)
    W_high = torch.mul(even_cols - odd_cols, norm_factor)

    return W_low, W_high


@torch.no_grad()
def inverse_haar_wavelet_transform_row(W_low, W_high):
    """
    row-wise inverse Haar wavelet transform
    """
    n_rows, half_cols = W_low.shape
    full_cols = 2 * half_cols

    W_filled_haar = torch.zeros(
        (n_rows, full_cols), dtype=W_low.dtype, device=W_low.device
    )

    norm_factor = 0.70703125
    W_filled_haar[:, ::2] = torch.mul(W_low + W_high, norm_factor)
    W_filled_haar[:, 1::2] = torch.mul(W_low - W_high, norm_factor)

    return W_filled_haar


@torch.no_grad()
def haar_wavelet_transform_col_low(W):
    """
    Applies a col-wise Haar wavelet transform to the input matrix and returns only the low-frequency component (W_low).
    """
    n_rows, _ = W.shape
    if n_rows % 2 != 0:
        raise ValueError("The number of rows must be even for Haar transform.")

    even_rows = W[::2, :]
    odd_rows = W[1::2, :]

    norm_factor = 0.70710677
    W_low = torch.mul(even_rows + odd_rows, norm_factor)

    return W_low


@torch.no_grad()
def haar_wavelet_transform_col_high(W):
    """
    Applies a col-wise Haar wavelet transform to the input matrix and returns only the high-frequency component (W_high).
    """
    n_rows, _ = W.shape
    if n_rows % 2 != 0:
        raise ValueError("The number of rows must be even for Haar transform.")

    even_rows = W[::2, :]
    odd_rows = W[1::2, :]

    norm_factor = 0.70710677
    W_high = torch.mul(even_rows - odd_rows, norm_factor)

    return W_high


@torch.no_grad()
def haar_wavelet_transform_col(W):
    """
    col-wise Haar wavelet transform
    """
    n_rows, _ = W.shape
    if n_rows % 2 != 0:
        raise ValueError("The number of rows must be even for Haar transform.")

    even_rows = W[::2, :]
    odd_rows = W[1::2, :]

    norm_factor = 0.70710677
    W_low = torch.mul(even_rows + odd_rows, norm_factor)
    W_high = torch.mul(even_rows - odd_rows, norm_factor)

    return W_low, W_high


@torch.no_grad()
def inverse_haar_wavelet_transform_col(W_low, W_high):
    """
    col-wise inverse Haar wavelet transform
    """
    half_rows, n_cols = W_low.shape
    full_rows = 2 * half_rows

    W_filled_haar = torch.zeros(
        (full_rows, n_cols), dtype=W_low.dtype, device=W_low.device
    )

    norm_factor = 0.70703125
    W_filled_haar[::2, :] = torch.mul(W_low + W_high, norm_factor)
    W_filled_haar[1::2, :] = torch.mul(W_low - W_high, norm_factor)

    return W_filled_haar

