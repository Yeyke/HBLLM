import torch
from utils.autosearch import structural_searching

"""
Used to generate masks for minor structural 2-bit salient data and split major 1-bit normal data according to different metric.
"""
def structural_guassian_distribution(tmp, H=None, metric="l2", up_lim=30):
    target_weights = tmp ** 2 / (torch.diag(H).reshape((1, -1))) ** 2
    mask1 = structural_searching(target_weights, up_lim, metric)
    mask2 = ~mask1
    print(mask1.sum() / mask1.numel(), mask2.sum() / mask2.numel())
    return mask1, mask2





