#!/usr/bin/env python

import torch


class Surrogate(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.A1 = torch.nn.Parameter(torch.tensor(5e-3, dtype=dtype))
        self.A2 = torch.nn.Parameter(torch.tensor(2e-3, dtype=dtype))
        self.sy = torch.nn.Parameter(torch.tensor(1000.0, dtype=dtype))
        self.eta = torch.nn.Parameter(torch.tensor(10.0, dtype=dtype))
        self.n = torch.nn.Parameter(torch.tensor(3.0, dtype=dtype))
        self.Q = torch.nn.Parameter(torch.tensor(5e4, dtype=dtype))
        self.R = torch.nn.Parameter(torch.tensor(8.3145, dtype=dtype))
        self.G0 = torch.nn.Parameter(torch.tensor(3e-3, dtype=dtype))
        self.C0 = torch.nn.Parameter(torch.tensor(4e-3, dtype=dtype))

    def forward(self, x):
        s = x[..., 0]
        T = x[..., 1]
        G = x[..., 2]
        C = x[..., 3]

        f = s - self.sy
        Hf = (torch.sign(f) + 1.0) / 2.0
        rep = (torch.abs(f) / self.eta) ** self.n * Hf
        rG = torch.exp(-G / self.G0)
        rC = torch.exp(-C / self.C0)
        aT = torch.exp(-self.Q / self.R / T)

        ep_dot = aT * (rG + rC) * rep
        G_dot = self.A1 + rG
        C_dot = self.A2 + rC

        return torch.stack([ep_dot, G_dot, C_dot], dim=-1)


if __name__ == "__main__":
    surrogate = torch.jit.script(Surrogate(torch.float64))
    torch.jit.save(surrogate, "surrogate.pt")
