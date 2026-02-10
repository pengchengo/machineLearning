import random
from sympy.geometry.entity import x
import torch
import torch.nn as nn
import torch.nn.functional as F

def corr2d(X,K): #@save
    h,w = K.shape
    Y = torch.zeros((X.shape[0]-h+1,X.shape[1]-w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+h,j:j+w]*K).sum()
    return Y