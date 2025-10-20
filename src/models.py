import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self, beta, n_neurons=64):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, n_neurons),
            nn.Softplus(beta), # default:torch.nn.Softplus(beta=1.0, threshold=20)
            nn.Linear(n_neurons, 1)
        )

    def forward(self, x):
        return self.layers(x)

class SimpleMLP2D(nn.Module):
    def __init__(self, beta, n_neurons=64):
        super(SimpleMLP2D, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, n_neurons),
            nn.Softplus(beta), # default:torch.nn.Softplus(beta=1.0, threshold=20)
            nn.Linear(n_neurons, 1)
        )

    def forward(self, x):
        return self.layers(x)