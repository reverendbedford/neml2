import torch
import neml2
import time


def eval(model, x):
    model.reinit([20, 50], 1, torch.device("cpu"))
    return model.value(x)


model = neml2.reload_model(
    "tests/unit/models/chaboche.i",
    "implicit_rate",
    enable_AD=False,
)
model.reinit([20, 50], 1, torch.device("cpu"))


f = torch.jit.trace(
    lambda x: eval(model, x), torch.rand(20, 50, model.input_axis().storage_size())
)

N = 1000
x = torch.rand(N, 20, 50, model.input_axis().storage_size())

t0 = time.time()
for i in range(N):
    y = model.value(x[i])
t1 = time.time()
print("original:", t1 - t0)

t0 = time.time()
with torch.inference_mode():
    for i in range(N):
        y = f(x[i])
t1 = time.time()
print("jit:     ", t1 - t0)
