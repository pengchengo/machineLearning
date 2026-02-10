import random
from sympy.geometry.entity import x
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))

net = MLP()
X = torch.randn(2, 20)
print(net(X))

class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self._modules[str(idx)] = module
    
    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X

net2 = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
print(net2(X))

class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand(20, 20, requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        X = self.linear(X)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()

net3 = FixedHiddenMLP()

net3.load_state_dict(torch.load('fixed_hidden_mlp.params'))

print(net3(X))