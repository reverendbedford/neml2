import torch
import pdb

torch.set_default_dtype(torch.float64)


class NonlinearSystem:
    # input
    s: torch.Tensor
    sn: torch.Tensor
    t: torch.Tensor
    tn: torch.Tensor

    # output
    output: torch.Tensor
    r: torch.Tensor

    # derivative
    deriv: torch.Tensor
    dr_ds: torch.Tensor
    dr_dsn: torch.Tensor
    dr_dt: torch.Tensor
    dr_dtn: torch.Tensor

    # implicit system
    x: torch.Tensor
    r: torch.Tensor
    J: torch.Tensor

    # parameter
    a: torch.Tensor

    def __init__(self, batch_shape):
        self.output = torch.zeros(batch_shape + (1,))
        self.deriv = torch.zeros(batch_shape + (1, 4))

    def setup_views(self, parent):
        self.parent = parent

        self.s = parent.input[..., 0].unsqueeze(-1)
        self.sn = parent.input[..., 1].unsqueeze(-1)
        self.t = parent.input[..., 2].unsqueeze(-1)
        self.tn = parent.input[..., 3].unsqueeze(-1)

        self.r = self.output[..., 0].unsqueeze(-1)
        self.dr_ds = self.deriv[..., 0, 0].unsqueeze(-1).unsqueeze(-1)
        self.dr_dsn = self.deriv[..., 0, 1].unsqueeze(-1).unsqueeze(-1)
        self.dr_dt = self.deriv[..., 0, 2].unsqueeze(-1).unsqueeze(-1)
        self.dr_dtn = self.deriv[..., 0, 3].unsqueeze(-1).unsqueeze(-1)

        self.x = parent.input[..., 0].unsqueeze(-1)
        self.r = self.output[..., 0].unsqueeze(-1)
        self.J = self.deriv[..., 0, 0].unsqueeze(-1).unsqueeze(-1)

    def forward(self):
        dt = self.t - self.tn
        sr = self.a * self.s**2
        self.r[:] = self.s - self.sn - sr * dt
        self.dr_ds[:] = (1 - 2 * self.a * self.s * dt).unsqueeze(-1)
        self.dr_dsn[:] = torch.tensor(-1.0)
        self.dr_dt[:] = -sr.unsqueeze(-1)
        self.dr_dtn[:] = sr.unsqueeze(-1)

    def assemble(self) -> (torch.Tensor, torch.Tensor):
        self.forward()
        return self.r.clone(), self.J.clone()

    def solution(self) -> torch.Tensor:
        return self.x.clone()

    def set_solution(self, x):
        self.x.data.copy_(x)


class Solver:
    def converged(self, i, nR, nR0) -> bool:
        print("ITERATION {:3d}, |R| = {:.3E}".format(i, torch.max(nR).item()))
        return torch.all(nR < 1e-10) or torch.all(nR / nR0 < 1e-8)

    def update(self, system: NonlinearSystem, xtr, R, J):
        dx = -torch.linalg.solve(J, R)
        xtr += dx
        system.set_solution(xtr)

    def solve(self, system: NonlinearSystem, xtr):
        for i in range(20):
            R, J = system.assemble()
            nR = torch.linalg.vector_norm(R, 2, -1, False)
            if i == 0:
                nR0 = nR

            if self.converged(i, nR, nR0):
                xtr.detach_()
                xtr += -torch.linalg.solve(J, R)
                return

            self.update(system, xtr, R, J)

        raise RuntimeError("Nonlinear solve failed")


class ImplicitUpdate:
    # input
    input: torch.Tensor
    s: torch.Tensor
    sn: torch.Tensor
    t: torch.Tensor
    tn: torch.Tensor

    # output
    output: torch.Tensor
    s: torch.Tensor

    # The implicit model
    model: NonlinearSystem

    # The solver
    solver: Solver

    def __init__(self, batch_shape, model, solver, input):
        self.output = torch.zeros(batch_shape + (1,))
        self.model = model
        self.solver = solver
        self.input = input

    def setup_views(self):
        self.s = self.input[..., 0].unsqueeze(-1)
        self.sn = self.input[..., 1].unsqueeze(-1)
        self.t = self.input[..., 2].unsqueeze(-1)
        self.tn = self.input[..., 3].unsqueeze(-1)

        self.snext = self.output[..., 0].unsqueeze(-1)

        self.model.setup_views(self)

    def forward(self):
        x = self.model.solution()
        self.solver.solve(self.model, x)
        self.output.copy_(x)


if __name__ == "__main__":
    batch_shape = (10,)
    input = torch.tensor([0.0, 1.0, 1.3, 1.1]).expand(batch_shape + (4,)).contiguous()
    solver = Solver()
    rate = NonlinearSystem(batch_shape)
    model = ImplicitUpdate(batch_shape, rate, solver, input)
    model.setup_views()

    rate.a = torch.full(batch_shape + (1,), 1.0, requires_grad=True)
    model.forward()
    # sol = model.output.detach()
    # rate.set_solution(sol)
    # r, J = rate.assemble()
    # sol += -torch.linalg.solve(J, r)
    # model.output.copy_(sol)
    g = torch.autograd.grad(
        model.snext, rate.a, grad_outputs=torch.ones_like(model.snext)
    )[0]
    print("AD", g)

    rate.a = torch.full(batch_shape + (1,), 1.0)
    model.forward()
    s0 = model.snext.clone()

    rate.a = torch.full(batch_shape + (1,), 1 + 1e-6)
    model.forward()
    s1 = model.snext.clone()

    print("finite differencing", (s1 - s0) / 1e-6)
