import torch.nn as nn


def MSE_Loss(output, target):
    return nn.MSELoss(output, target)

def L1_Loss(output, target):
    return nn.L1Loss(output, target)

def Smooth_L1_loss(output, target):
    return nn.SmoothL1Loss(output, target)