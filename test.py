import torch
import torch.nn as nn
from functorch import jacrev, vmap


class Lin(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, x):
        return torch.einsum("...ij,...j", self.weight, x) + self.bias


if __name__ == "__main__":
    nbatch = 100
    x = torch.rand(nbatch, 25)
    net = Lin(torch.rand(10, 25), torch.rand(10))

    print("Input batched, parameters unbatch")
    print("Input shape: %s" % str(x.shape))
    print("Output shape: %s" % str(net(x).shape))
    print("Jacobian shape: %s" % str(vmap(jacrev(net))(x).shape))

    print("Input batched, parameters batched")
    x = torch.rand(nbatch, 25)
    net = Lin(torch.rand(nbatch, 10, 25), torch.rand(nbatch, 10))
    print("Input shape: %s" % str(x.shape))
    print("Output shape: %s" % str(net(x).shape))
    print("Jacobian shape: %s" % str(vmap(jacrev(net))(x).shape))
