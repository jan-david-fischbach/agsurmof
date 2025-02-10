# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import numpy as np
from surmof_cavity import ag_surmof_cavity_det_smat as det_smat
import matplotlib.pyplot as plt
import matplotlib as mpl
# %config InlineBackend.figure_format='retina'
from functools import partial
from diffaaable.selective import selective_refinement_aaa
import pickle
from tqdm import tqdm


# %%
def find_qnms(ts, npoles=3, osc_strength=1, damping=1, checkpointing=True):
  """Find the poles of the S-matrix of a surmof cavity 

  Args:
    ts (iterable): Cavity Thicknesses to scan over
    npoles (int, optional): 
        Number of Poles of the SURMOF material to consider. Defaults to 3.
    osc_strength (int, optional): 
        Scaling factor the oscillator strength. Defaults to 1.
    damping (int, optional): 
        Scaling factor for the damping. Defaults to 1.
    checkpointing (bool, optional): 
        Whether to write results to a cache file (and avoid recomputing 
        results already present in the future). Defaults to True.
  """
  fname = f"tmp/fine_{npoles}pole_{osc_strength}osc_{damping}damping.pkl"

  try:
    with open(fname, 'rb') as file:
      cache = pickle.load(file)
    all_poles    = cache['poles']
    all_residues = cache['residues']

    ts_new = [t for t in ts if t not in cache['thickness']] # TODO
  except FileNotFoundError:
    all_poles = []
    all_residues = []
    ts_new = ts

  for thickness in tqdm(ts_new):
    f = partial(det_smat, 
      thickness=thickness, npoles=npoles, 
      scale_osc=osc_strength, scale_damping=damping
    )

    poles, residues, evals = selective_refinement_aaa(
      f, domain=[1-0.5j, 2.5+0.05j], 
      N=400, use_adaptive=False, tol_pol=1e-6, Dmax=11)
                    
    all_poles.append(poles)
    all_residues.append(residues)

    ts_tmp = ts[:len(all_poles)]
    sorter = np.argsort(ts_tmp)
    poles_tmp    = [all_poles[i] for i in sorter]
    residues_tmp = [all_residues[i] for i in sorter]
    ts_tmp       = [ts_tmp[i] for i in sorter]
    with open(fname, "wb") as file:
      pickle.dump({
        "poles":    poles_tmp, 
        "residues": residues_tmp, 
        "thickness":ts_tmp
      }, file)
  
  return all_poles, all_residues


# %%
def eyes(ts, all_poles, all_residues):
  cmap = mpl.cm.viridis
  norm = mpl.colors.Normalize(vmin=min(ts), vmax=max(ts))
  colors = cmap(norm(ts))
  for color, poles, residues in zip(colors, all_poles, all_residues):
      filtered_poles = poles
      plt.scatter(filtered_poles.real, filtered_poles.imag, color=color, s=np.sqrt(np.abs(residues))*100)

  plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), ax=plt.gca(), label="cavity thickness [$\mu m$]")

  plt.xlabel("$\Re\{\hbar \omega\}$ [eV]")
  plt.ylabel("$\Im\{\hbar \omega\}$ [eV]")


# %%
if __name__ == "__main__":
  ts = 0.005*(np.arange(1, 120)+1)

  all_poles, all_residues = find_qnms(ts, npoles=1, osc_strength=1, damping=1, checkpointing=True)
  plt.figure()
  eyes(ts, all_poles, all_residues)
  plt.savefig("tmp/prelim_1pole_1osc_1damp.png", dpi=600)
  

# %%
if __name__ == "__main__":
  all_poles, all_residues = find_qnms(ts, npoles=3, osc_strength=1, damping=1, checkpointing=True)
  plt.figure()
  eyes(ts, all_poles, all_residues)
  plt.savefig("tmp/prelim_3pole_1osc_1damp.png", dpi=600)

# %%
if __name__ == "__main__":
  for osc in [1, 0.25, 0.1, 0.025, 0.1]:
    all_poles, all_residues = find_qnms(ts, npoles=1, osc_strength=osc, damping=1, checkpointing=True)
    plt.figure()
    eyes(ts, all_poles, all_residues)
    plt.savefig(f"tmp/prelim_1pole_{osc}osc_1damp.png", dpi=600)
