import torch
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import cm, colors

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 24

plt.rcParams["text.usetex"] = True
plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


res = torch.jit.load(
    "tests/regression/solid_mechanics/viscoplasticity/misc/torch_script/gold/result.pt"
)
res = dict(res.named_buffers())

strain = res["input.forces/E"][..., 0]
T = res["input.forces/T"][..., 0]
stress = res["output.state/S"][..., 0]
G = res["output.state/G"][..., 0]
C = res["output.state/C"][..., 0]

cs = np.linspace(0, 1, 20)
norm = colors.Normalize(vmin=0, vmax=1)
sm = cm.ScalarMappable(norm=norm, cmap="turbo")

fig, ax = plt.subplots()
for i, c in enumerate(cs):
    ax.plot(strain[..., i], stress[..., i], color=sm.to_rgba(c))
ax.set_xlabel("Strain")
ax.set_ylabel("Stress")
fig.tight_layout()
fig.savefig("rom_stress.png")
fig.savefig("rom_stress.pdf")

fig, ax = plt.subplots()
for i, c in enumerate(cs):
    ax.plot(strain[..., i], G[..., i], color=sm.to_rgba(c))
ax.set_xlabel("Strain")
ax.set_ylabel("Grain size")
fig.tight_layout()
fig.savefig("rom_grain.png")
fig.savefig("rom_grain.pdf")

fig, ax = plt.subplots()
for i, c in enumerate(cs):
    ax.plot(strain[..., i], C[..., i], color=sm.to_rgba(c))
ax.set_xlabel("Strain")
ax.set_ylabel("Stoichiometry")
fig.tight_layout()
fig.savefig("rom_st.png")
fig.savefig("rom_st.pdf")
