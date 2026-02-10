import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])