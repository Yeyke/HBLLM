import torch.nn as nn
from utils.haar_search import *
from utils.split_search import fill_significant_columns_with_avg, separate_columns_withnan


class HaarBinarization(nn.Module):
    def __init__(self, weight, groupsize=-1, method="col-hbraq", group_partition="global", share_mean=False):
        super().__init__()
        oc, ic = weight.shape
        if groupsize == -1:
            groupsize = ic
        self.groupsize = groupsize
        assert method in ["col-hbraq", "row-hbraq"], f"Unsupported method: {method}"
        assert group_partition in ["global", "row"], f"Unsupported group_partition: {group_partition}"
        self.method = method
        self.group_partition = group_partition
        self.share_mean = share_mean

    def quantize(self, W1, salient_mask, unsalient_mask):
        if self.method == "row-hbraq":
            unsalient_block, salient_block = fill_significant_columns_with_avg(
                W1, salient_mask, unsalient_mask
            )
            if self.group_partition == "global":
                return BiLLM_with_row_haar_global(
                    unsalient_block, salient_block
                )
            elif self.group_partition == "row":
                if self.share_mean:
                    return BiLLM_with_row_haar_row_shared_mean(
                        unsalient_block, salient_block
                    )
                else:
                    return BiLLM_with_row_haar_row(
                        unsalient_block, salient_block
                    )

        elif self.method == "col-hbraq":
            unsalient_block, salient_block = separate_columns_withnan(
                W1, salient_mask, unsalient_mask
            )
            if self.group_partition == "global":
                return BiLLM_with_col_haar_global(
                    W1, salient_mask, unsalient_block, salient_block
                )
            elif self.group_partition == "row":
                if self.share_mean:
                    return BiLLM_with_col_haar_row_shared_mean(
                        W1, salient_mask, unsalient_block, salient_block
                    )
                else:
                    return BiLLM_with_col_haar_row(
                        W1, salient_mask, unsalient_block, salient_block
                    )

        else:
            raise NotImplementedError(f"Unknown method: {self.method}")
