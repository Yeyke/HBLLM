import torch
index = 0

@torch.no_grad()
def high_order_residual_rms(x, mask, order=1):
    if x.dim() == 1:
        x = x.unsqueeze(0)

    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    global index
    index += 1
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float("nan")))

        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(
            torch.isnan(mean_tensor_all),
            torch.zeros_like(mean_tensor_all),
            mean_tensor_all,
        )

        masked_x_tensor -= mean_tensor_all[:, None]

        valid_counts = torch.sum(mask, dim=1).float()
        scale_tensor_all = torch.sqrt(
            torch.nansum(masked_x_tensor**2, dim=1) / valid_counts
        )

        scale_tensor_all = torch.where(
            torch.isnan(scale_tensor_all),
            torch.zeros_like(scale_tensor_all),
            scale_tensor_all,
        )

        binary = torch.sign(masked_x_tensor)
        binary *= scale_tensor_all[:, None]
        binary += mean_tensor_all[:, None]
        sum_order = sum_order + binary * mask

    if sum_order.shape[0] == 1:
        sum_order = sum_order.squeeze(0)

    return sum_order

@torch.no_grad()
def high_order_residual(x, mask, order=1):
    if x.dim() == 1:
        x = x.unsqueeze(0)

    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    global index
    index += 1
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float("nan")))

        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(
            torch.isnan(mean_tensor_all),
            torch.zeros_like(mean_tensor_all),
            mean_tensor_all,
        )
        masked_x_tensor -= mean_tensor_all[:, None]
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(
            torch.isnan(scale_tensor_all),
            torch.zeros_like(scale_tensor_all),
            scale_tensor_all,
        )

        binary = torch.sign(masked_x_tensor)
        binary *= scale_tensor_all[:, None]
        binary += mean_tensor_all[:, None]
        sum_order = sum_order + binary * mask

    if sum_order.shape[0] == 1:
        sum_order = sum_order.squeeze(0)

    return sum_order

@torch.no_grad()
def high_order_residual_rms_shared_mean(
    x, mask, mean_tensor_all
):
    if x.dim() == 1:
        x = x.unsqueeze(0)

    sum_order = torch.zeros_like(x)
    new_matrix = x * mask
    device = x.device

    masked_x_tensor = torch.where(
        mask, new_matrix, torch.tensor(float("nan"), device=device)
    )
    mean_tensor_all = torch.where(
        torch.isnan(mean_tensor_all),
        torch.zeros_like(mean_tensor_all),
        mean_tensor_all,
    )
    masked_x_tensor -= mean_tensor_all[:, None]

    valid_counts = torch.sum(mask, dim=1).float()
    scale_tensor_all = torch.sqrt(
        torch.nansum(masked_x_tensor**2, dim=1) / valid_counts
    )

    scale_tensor_all = torch.where(
        torch.isnan(scale_tensor_all),
        torch.zeros_like(scale_tensor_all, device=device),
        scale_tensor_all,
    )

    scale_mask = torch.sign(masked_x_tensor)
    scale_mask[~mask] = float("nan")

    binary = (
        torch.sign(masked_x_tensor) * scale_tensor_all[:, None]
        + mean_tensor_all[:, None]
    )
    sum_order += binary * mask
    if sum_order.shape[0] == 1:
        sum_order = sum_order.squeeze(0)
        scale_mask = scale_mask.squeeze(0)
        scale_tensor_all = scale_tensor_all.squeeze(0)

    return sum_order
