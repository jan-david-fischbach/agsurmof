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
from qnmsc.cylinder import det_tmat
import matplotlib.pyplot as plt
import matplotlib as mpl
# %config InlineBackend.figure_format='retina'
from functools import partial
from diffaaable.selective import selective_refinement_aaa
import pickle
from tqdm import tqdm
from pathlib import Path

# %%
def filename(npoles, osc_strength, damping, domain):
  d = domain
  folder = Path(f"tmp/elli/domain_{d[0].real}_{d[0].imag}_{d[1].real}_{d[1].imag}")
  folder.mkdir(parents=True, exist_ok=True)
  file = folder/f"fine_{npoles}pole_{osc_strength}osc_{damping}damping.pkl"
  return file

def find_qnms(rs, betas, npoles=3, osc_strength=1, damping=1, 
              domain=[1-0.5j, 2.5+0.05j], checkpointing=True, plotting=True):
  """Find the poles of the S-matrix of a surmof cavity 

  Args:
    rs (iterable): Cylinder Radius to scan over
    betas (iterable): Beta values to scan over (has to have the same size as `rs`)
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
  if len(rs) != len(betas):
    raise ValueError("The number of radii and betas has to be the same.")

  fname = filename(npoles, osc_strength, damping, domain)
  ps = list(zip(rs, betas))
  all_poles = []
  all_residues = []
  ps_new = ps
  if checkpointing:
    try:
      with open(fname, 'rb') as file:
        cache = pickle.load(file)
      all_poles    = cache['poles']
      all_residues = cache['residues']

      ps_new = [p for p in ps if p not in cache['param']]
      if len(ps_new) == 0:
        ps = np.array(cache['param'])
      else:
        ps = np.concat([np.array(cache['param']), np.array(ps_new)])
    except FileNotFoundError:
      pass

  for parameter in tqdm(ps_new):
    radius, beta = parameter
    f = partial(det_tmat, 
      radius=radius, beta=beta, npoles=npoles, 
      scale_osc=osc_strength, scale_damping=damping
    )

    poles, residues, evals = selective_refinement_aaa(
      f, domain=domain, 
      N=100, use_adaptive=False, tol_pol=1e-7, Dmax=22)
                    
    all_poles.append(poles)
    all_residues.append(residues)

    if checkpointing:
      ps_tmp = ps[:len(all_poles)]
      sort_by = [np.sum(p) for p in ps_tmp]
      sorter = np.argsort(sort_by)
      poles_tmp    = [all_poles[i] for i in sorter]
      residues_tmp = [all_residues[i] for i in sorter]
      ps_tmp       = [ps_tmp[i] for i in sorter]
      with open(fname, "wb") as file:
        pickle.dump({
          "poles":    poles_tmp, 
          "residues": residues_tmp, 
          "param": ps_tmp
        }, file)

  if plotting:
    plt.figure()
    eyes(rs, all_poles, all_residues)
    plt.savefig(fname.parent/(fname.stem+".png"), dpi=600)
    plt.close()
  
  return all_poles, all_residues

def track_qnms(poles, residues, threshold = 1):
  res_prev = residues[0]
  pol_prev = poles[0]
  mapping_prev = np.arange(len(pol_prev))

  modes = []
  max_mode_idx = mapping_prev[-1] if len(mapping_prev) else -1

  for pol, res in zip(poles, residues):
    connection_matrix = np.abs(res_prev[:, None]-res[None, :]) / np.abs(res_prev[:, None]+res[None, :])
    connection_matrix_pol = np.abs(pol_prev[:, None]-pol[None, :]) / np.abs(pol_prev[:, None]+pol[None, :]) / np.abs(res_prev[:, None]+res[None, :])
    conny = 1/(connection_matrix+connection_matrix_pol+1e-20) # high values -> strong connection

    new_mapping = np.empty(len(pol), dtype=int)
    new_mapping[:] = -1
    while np.any(conny>threshold):
      conn_from, conn_to = np.unravel_index(conny.argmax(), conny.shape)
      new_mapping[conn_to] = mapping_prev[conn_from]

      conny[conn_from, :] = 0 
      conny[:, conn_to] = 0 
    
    for i, val in enumerate(new_mapping):
      if val<0:
        max_mode_idx += 1
        new_mapping[i] = max_mode_idx

    modes.append(new_mapping)
    mapping_prev = new_mapping

    res_prev = res
    pol_prev = pol

  poles_tracked = np.empty((len(poles), max_mode_idx+1), dtype=complex)
  poles_tracked[:] = np.nan
  residues_tracked = poles_tracked.copy()

  for i, (pol, res, mod) in enumerate(zip(poles, residues, modes)):
    for p, r, m in zip(pol, res, mod):
      poles_tracked[i, m] = p 
      residues_tracked[i, m] = r

  poles_tracked_filtered = poles_tracked.copy()
  poles_tracked_filtered[np.abs(residues_tracked)<1e-5] = np.nan

  filter = np.sum(~np.isnan(poles_tracked), axis=0) > 5

  return poles_tracked_filtered[:, filter], residues_tracked[:, filter]

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
  rs = 0.003*(np.arange(1, 30)+1)
  betas = 0.7*np.ones_like(rs)

  domain = [1-0.5j, 2.5+0.05j]

  all_poles, all_residues = find_qnms(
    rs, betas, npoles=3, osc_strength=1, damping=1, domain=domain, 
    checkpointing=True
  )


  betas = 0.6 + 0.01*np.arange(0, 21)
  rs = 0.03*np.ones_like(betas)

  domain = [1-0.5j, 2.5+0.05j]

  all_poles, all_residues = find_qnms(
    rs, betas, npoles=3, osc_strength=1, damping=1, domain=domain, 
    checkpointing=True
  )