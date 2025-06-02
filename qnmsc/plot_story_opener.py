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
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
from qnmsc.plot_splitting import plot_splitting
from qnmsc.plot_trajectories import load_data, track_qnms, select_modes
from qnmsc import inverse_eigenproblem
import matplotlib.pyplot as plt
import numpy as np

# %%
domain = [0.7-0.7j, 5.0+0.05j]
domain = [1-0.5j, 2.5+0.05j]

npoles = 3
scale_osc = 1
scale_damping = 1

modenumber = 1

# %%
poles, residues, thickness, material_poles = load_data(
  npoles, scale_osc, scale_damping, domain
)
poles_tracked, residues_tracked = track_qnms(poles, residues)

selection, modenumber = select_modes(npoles, scale_osc, scale_damping, domain, modenumber=modenumber)
selected = poles_tracked[:, selection]

# %%
om_os = []
for p in selected:
  om_o, *couplings = inverse_eigenproblem.solve_inv_eig(
            p, material_poles
        )

  Cs = np.array(couplings)
  iVii = Cs / (-1* material_poles)

  corr = np.sum(iVii, axis=-1)

  om_os.append(om_o + corr)

# %%
plt.plot(thickness, selected.real)
plt.plot(thickness, om_os)
plt.xlim(0.05, 0.4)
plt.ylim()

# %%
