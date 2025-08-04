# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: qnmsc (3.11.10)
#     language: python
#     name: python3
# ---

# %%
import pickle
import numpy as np
import qnmsc.track_qnms
import matplotlib.pyplot as plt
# %matplotlib widget

# %% [markdown]
# ## Load Data

# %%
fname = "assets/in/fine_3pole_1osc_1damping_beta.pkl"
with open(fname, "rb") as file:
    data = pickle.load(file)

poles = data["poles"]
residues = data["residues"]
beta = np.array(data['param'])[:, 1]

# %% [markdown]
# ## View Poles

# %%
plt.figure()
for ps, b in zip(poles, beta):
    plt.scatter([b]*len(ps), ps.real)

# %% [markdown]
# ## Select betas:

# %%
idx = np.argmax(beta > 0.4)
poles = poles[idx:]
residues = residues[idx:]
beta = beta[idx:]

# %% [markdown]
# ## Track Poles

# %%
tracked_poles, tracked_residues = qnmsc.track_qnms.track_qnms(poles=poles, residues=residues, threshold=0.1)

# %% [markdown]
# ## View Tracking Results:

# %%
plt.figure()
for pole in tracked_poles.T:
  plt.plot(beta, pole.real)

for ps, b in zip(poles, beta):
  plt.scatter([b]*len(ps), ps.real, color="gray", alpha=0.5)

plt.xlabel(r"$\beta$")
plt.ylabel(r"$\Re\{\hbar \omega\}$ [eV]")

plt.savefig("assets/out/connected_real.pdf")

# %%
plt.figure()
vmin = min(beta)
vmax = max(beta)
for pole in tracked_poles.T:
  plt.scatter(pole.real, pole.imag, c=beta, cmap='viridis', vmin=vmin, vmax=vmax)
  plt.plot(pole.real, pole.imag, color='gray', alpha=0.5)
plt.colorbar()

# for ps, b in zip(poles, beta):
#   plt.scatter(ps.real, ps.imag, color="k", marker="x", alpha=0.5)

plt.ylabel(r"$\Im\{\hbar \omega\}$ [eV]")
plt.xlabel(r"$\Re\{\hbar \omega\}$ [eV]")

plt.xlim(1.64877, 1.9303), plt.ylim(-0.013229, -0.010736)
plt.savefig("assets/out/connected_complex_plane.pdf")

# %% [markdown]
# ## Save Results

# %%
with open("assets/out/tracked_3pole_1osc_1damping_beta.pkl", "wb") as file:
    pickle.dump({"poles": tracked_poles, "residues": tracked_residues, "param": beta}, file)
