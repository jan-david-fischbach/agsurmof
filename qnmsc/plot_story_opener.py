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
from qnmsc.plot_trajectories import load_data, track_qnms, select_modes, to_eV, calc_material_poles
from qnmsc import inverse_eigenproblem
import matplotlib.pyplot as plt
import numpy as np

# %%
from qnmsc.mpl_config import mm
import qnmsc.mpl_config
qnmsc.mpl_config.config()

# %%
domain = [0.7-0.7j, 5+0.05j]

npoles = 3
scale_osc = 0.1
scale_damping = 1


# %%
def plot_comic(modenumber, npoles, scale_osc, scale_damping, domain, xlim=None, ylim=None):
  poles, residues, thickness, material_poles = load_data(
    npoles, scale_osc, scale_damping, domain
  )
  for i, matpole in enumerate(material_poles):
    plt.axhline(matpole.real, color=f"C{i}")

  poles_tracked, residues_tracked = track_qnms(poles, residues)

  selection, modenumber = select_modes(npoles, scale_osc, scale_damping, domain, modenumber=modenumber)
  selected = poles_tracked[:, selection]

  om_os = []
  for p in selected:
    om_o, *couplings = inverse_eigenproblem.solve_inv_eig(
              p, material_poles
          )

    Cs = np.array(couplings)
    iVii = Cs / (-1* material_poles)

    corr = np.sum(iVii, axis=-1)

    om_os.append(om_o + corr)
  
  om_os = np.array(om_os)
  plt.plot(thickness, selected.real, color="k")

  idx_last = np.argmin(np.abs(thickness - xlim[1]))
  for i, line in enumerate(selected.real.T):
    cent = material_poles.real[0]
    offset = 0.04 if i==3 else 0
    plt.text(thickness[idx_last]*1.01, (line[idx_last]-cent)*1.06 + cent + offset -0.04, f'm{i+1}')

  plt.plot(thickness, om_os.real, color="gray")
  plt.xlim(xlim)
  plt.ylim(ylim)

  return thickness, om_os

# %%
from PIL import Image

def plot_cartoon(ax, path):
  """
  This function uses the cairosvg library to convert an SVG file to a PNG image and then displays it using matplotlib.

  Args:
      ax (Axes): The matplotlib Axes object where the image will be displayed.
      path (str, optional): File path of the svg file
  """

  # Load PNG image
  image = Image.open(path)
  # Display using matplotlib

  ax.imshow(image)
  ax.axis('off')
  ax.set_xlim(300, 1550)


# %%
from qnmsc.materials import eps_surmof



# %%
fig, axs = plt.subplots(3, 3, figsize=(90*mm, 70*mm), sharex="col", sharey="col", constrained_layout=True, width_ratios=[1,1,1.5])

ylim = [1.4, 2.1]
xlim = [0.4, 0.6]
plt.sca(axs[1,2])
thickness, om_os = plot_comic(2, 1, scale_osc, scale_damping, domain, xlim=xlim)
hbar_omega = np.linspace(*ylim, 601)

plt.sca(axs[0,2])
plt.plot(thickness, om_os.real, color="k")


plt.sca(axs[2,2])
plot_comic(2, 3, scale_osc, scale_damping, domain, xlim=xlim, ylim=ylim)


npol = [0,1,3]
for i in range(3):
  plot_cartoon(axs[i, 0], f"assets/cavity-{i+1}.png")

  plt.sca(axs[i, 1])
  eps = eps_surmof(
        hbar_omega, 
        npoles=npol[i], 
        scale_osc=scale_osc, 
        scale_damping=scale_damping
    )

  lw = 0.7
  plt.plot(eps.real, hbar_omega, lw=lw, color="gray", label=r"$\Re\{\varepsilon\}$")
  plt.plot(eps.imag, hbar_omega, lw=lw, color="k", label=r"$\Im\{\varepsilon\}$")

  material_poles = to_eV(calc_material_poles(scale_osc, scale_damping))[:npol[i]]
  for j, matpole in enumerate(material_poles):
    plt.plot([0, eps.imag[np.argmin(np.abs(hbar_omega - matpole.real))]], [matpole.real]*2, color=f"C{j}", lw=lw)
  #plt.xlim((-0.5, 2))


plt.sca(axs[-1,-1])
plt.xlabel(r"System Parameter $\rho$")
plt.xticks([])
plt.yticks([])

plt.sca(axs[-1,-2])
#plt.xlabel(r"Susceptibility $\chi$")
plt.xlabel(r"Permittivity $\varepsilon$")
plt.xticks([])
plt.yticks([])
plt.ylim(ylim)

plt.sca(axs[1,1])
xlim=plt.xlim()
ylim=plt.ylim()
plt.xlim(xlim)
plt.ylim(ylim)

for ax in axs[:, 1]:
  ax.set_ylabel(r"Frequency $\omega$")
  ax.spines['left'].set_visible(False)
  ax.axvline(0, color="k", lw=0.5)
  #axs[i, -1].yaxis.set_label_position("right")
#fig.supylabel("Frequency $\omega$", x=1.04, ha="right")
axs[0,1].set_ylabel(r"Frequency $\omega$")
axs[0,1].legend(handlelength=1.2, fontsize=5, loc="upper right", framealpha=0.9, frameon=True)

# axs[0,0].set_title("Mechanical\nOscillators")
# axs[0,1].set_title("Drude-Lorentz\nOscillators")
# axs[0,2].set_title("Hybrid Modes")

for ax in axs.flatten():
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)

# fig.get_layout_engine().set(w_pad=4 / 72, h_pad=4 / 72, hspace=0.2,
#                             wspace=0)
for j, axj in enumerate(axs.T):
  for i, ax in enumerate(axj):
    letter = chr(ord("a")+i)
    x = 0.8 if j == 2 else 0.02
    alpha = 1 if j==0 else 0 
    ax.annotate(
          f" ({letter}{j+1})",
          xy=(x, 1), xycoords='axes fraction',
          xytext=(+0.5, -0.5), textcoords='offset fontsize',
          fontsize='medium', verticalalignment='top', fontfamily='serif',
          bbox=dict(facecolor=(1,1,1,alpha), edgecolor='none', pad=2.0))

plt.savefig("out/cartoon.pdf", bbox_inches="tight", dpi=1200)

# %%
