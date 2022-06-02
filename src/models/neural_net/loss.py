import torch.nn as nn


def MSE_Loss(output, target):
    return nn.MSELoss(output, target)
