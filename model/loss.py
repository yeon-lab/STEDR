import torch
import torch.nn.functional as F

def loss(y, t, y0_pred, y1_pred, t_pred):
    loss_t = F.binary_cross_entropy(t_pred, t)
    loss_y = torch.sum((1. - t) * torch.square(y - y0_pred)) + torch.sum(t * torch.square(y - y1_pred))
    return loss_y + loss_t
