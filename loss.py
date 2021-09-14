import torch
import torch.nn as nn


class CCCLoss(nn.Module):
    def __init__(self):
        super(CCCLoss, self).__init__()

    def forward(self, y_pred, y_true, seq_lens=None):
        if seq_lens is not None:
            mask = torch.ones_like(y_true, device=y_true.device)
            for i, seq_len in enumerate(seq_lens):
                mask[i, seq_len:] = 0
        else:
            mask = torch.ones_like(y_true, device=y_true.device)
        y_true_mean = torch.sum(y_true * mask, dim=1, keepdim=True) / torch.sum(mask, dim=1, keepdim=True)
        y_pred_mean = torch.sum(y_pred * mask, dim=1, keepdim=True) / torch.sum(mask, dim=1, keepdim=True)

        y_true_var = torch.sum(mask * (y_true - y_true_mean) ** 2, dim=1, keepdim=True) / torch.sum(mask, dim=1,
                                                                                                    keepdim=True)
        y_pred_var = torch.sum(mask * (y_pred - y_pred_mean) ** 2, dim=1, keepdim=True) / torch.sum(mask, dim=1,
                                                                                                    keepdim=True)

        cov = torch.sum(mask * (y_true - y_true_mean) * (y_pred - y_pred_mean), dim=1, keepdim=True) \
              / torch.sum(mask, dim=1, keepdim=True)

        ccc = torch.mean(2.0 * cov / (y_true_var + y_pred_var + (y_true_mean - y_pred_mean) ** 2), dim=0)
        ccc = ccc.squeeze(0)
        ccc_loss = 1.0 - ccc

        return ccc_loss