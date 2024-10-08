import torch
from torch.profiler import profile, record_function, ProfilerActivity
import neml2
import time


def eval(model, x):
    model.reinit(x.shape[:2], 0, torch.device("cpu"))
    return model.value(x)


model = neml2.reload_model(
    "tests/unit/models/chaboche.i",
    "implicit_rate",
    enable_AD=False,
)

f = torch.jit.trace(
    lambda x: eval(model, x),
    torch.rand(20, 50, model.input_axis().storage_size()),
    _force_outplace=False,
)

N = 1000
x = torch.rand(N, 20, 50, model.input_axis().storage_size())

t0 = time.time()
for i in range(N):
    model.reinit([20, 50], 0, torch.device("cpu"))
    y = model.value(x[i])
t1 = time.time()
print("original:", t1 - t0)

t0 = time.time()
with torch.inference_mode():
    for i in range(N):
        y = f(x[i])
t1 = time.time()
print("jit:     ", t1 - t0)

# with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
#     with record_function("original"):
#         for i in range(N):
#             model.value(x[i])
# print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=25))

# with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
#     with record_function("original"):
#         for i in range(N):
#             y = f(x[i])
# print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=25))
