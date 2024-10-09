import torch
from torch.profiler import profile, record_function, ProfilerActivity
import neml2
import time

torch.random.manual_seed(42)
device = torch.device("cuda")

model = neml2.reload_model(
    "tests/unit/models/chaboche.i",
    "implicit_rate",
    enable_AD=False,
)
model.reinit(device=device)

f = torch.jit.trace(
    model.value, torch.rand(1, 1, model.input_axis().storage_size(), device=device)
)

N = 1000
x = torch.rand(N, 20, 50, model.input_axis().storage_size(), device=device)

t0 = time.time()
for i in range(N):
    y = model.value(x[i])
t1 = time.time()
print("original:", t1 - t0)

t0 = time.time()
# with torch.inference_mode():
for i in range(N):
    y = f(x[i])
t1 = time.time()
print("jit:     ", t1 - t0)

# with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
#     with record_function("original"):
#         model.value(x[0])
# print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=25))

# with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
#     with record_function("original"):
#         y = f(x[0])
# print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=25))
