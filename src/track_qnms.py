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
from pathlib import Path

# %%
def find_qnms(ts, npoles=3, osc_strength=1, damping=1, 
              domain=[1-0.5j, 2.5+0.05j], checkpointing=True, plotting=True):
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
  d = domain
  folder = Path(f"tmp/domain_{d[0].real}_{d[0].imag}_{d[1].real}_{d[1].imag}")
  folder.mkdir(parents=True, exist_ok=True)
  fname = folder/f"fine_{npoles}pole_{osc_strength}osc_{damping}damping.pkl"

  all_poles = []
  all_residues = []
  ts_new = ts
  if checkpointing:
    try:
      with open(fname, 'rb') as file:
        cache = pickle.load(file)
      all_poles    = cache['poles']
      all_residues = cache['residues']

      ts_new = [t for t in ts if t not in cache['thickness']] # TODO
    except FileNotFoundError:
      pass

  for thickness in tqdm(ts_new):
    f = partial(det_smat, 
      thickness=thickness, npoles=npoles, 
      scale_osc=osc_strength, scale_damping=damping
    )

    poles, residues, evals = selective_refinement_aaa(
      f, domain=domain, 
      N=400, use_adaptive=False, tol_pol=1e-6, Dmax=11)
                    
    all_poles.append(poles)
    all_residues.append(residues)

    if checkpointing:
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
  if plotting:
    plt.figure()
    eyes(ts, all_poles, all_residues)
    plt.savefig(fname.parent/(fname.stem+".png"), dpi=600)
    plt.close()
  
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
  domain = [1-0.5j, 2.5+0.05j]

  for npoles in [1,3]:
    all_poles, all_residues = find_qnms(
      ts, npoles=npoles, osc_strength=1, damping=1, domain=domain, 
      checkpointing=True
    )

  for osc in [1, 0.25, 0.1, 0.025, 0.01]:
    all_poles, all_residues = find_qnms(
      ts, npoles=1, osc_strength=osc, damping=1, domain=domain, 
      checkpointing=True
    )

# %%
  ts = 0.005*(np.arange(1, 240)+1)
  domain = [0.7-0.7j, 5+0.05j]

  for npoles in [1,3]:
    all_poles, all_residues = find_qnms(
      ts, npoles=npoles, osc_strength=1, damping=1, domain=domain, 
      checkpointing=True
    )