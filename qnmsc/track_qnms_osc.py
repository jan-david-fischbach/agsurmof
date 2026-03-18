from qnmsc.materials import eps_surmof
from qnmsc.track_qnms import *

def filename_osc(npoles, ds, damping, kpar_eV, domain):
  d = domain
  folder = Path(f"tmp/domain_{d[0].real}_{d[0].imag}_{d[1].real}_{d[1].imag}")
  folder.mkdir(parents=True, exist_ok=True)
  file = folder/f"fine_{npoles}pole_{ds[0]}m_{ds[1]}s_{ds[2]}m_{kpar_eV}kp_{damping}damping.pkl"
  return file

def find_qnms_osc(osc_scalings, npoles=3, ds=(0.01, 1/4.780315, 0.03), damping=1, kpar_eV=0,
              domain=[1-0.5j, 2.5+0.05j], checkpointing=True, plotting=True, Dmax=22, tol_aaa=1e-9, N=100):
  """Find the poles of the S-matrix of a surmof cavity 

  Args:
    osc_scalings (iterable): Oscillator Strengths to scan over
    npoles (int, optional): 
        Number of Poles of the SURMOF material to consider. Defaults to 3.
    ds (tuple[int, int, int], optional): 
        thicknesses of mirror1, surmof, mirror2.
    damping (int, optional): 
        Scaling factor for the damping. Defaults to 1.
    checkpointing (bool, optional): 
        Whether to write results to a cache file (and avoid recomputing 
        results already present in the future). Defaults to True.
  """
  fname = filename_osc(npoles, ds, damping, kpar_eV, domain)

  all_poles = []
  all_residues = []
  oscs_new = osc_scalings
  if checkpointing:
    try:
      with open(fname, 'rb') as file:
        cache = pickle.load(file)
      all_poles    = cache['poles']
      all_residues = cache['residues']

      oscs_new = [osc for osc in osc_scalings if osc not in cache['osc_scaling']] # TODO
      osc_scalings = np.concat([np.array(cache['osc_scaling']), np.array(oscs_new)])
    except FileNotFoundError:
      pass

  for osc in tqdm(oscs_new):
    f = partial(det_smat, 
      thickness=ds[1], mirror1d=ds[0], mirror2d=ds[2], npoles=npoles, 
      scale_osc=osc, scale_damping=damping, kpar_eV=kpar_eV
    )

    poles, residues, evals = selective_refinement_aaa(
      f, domain=domain, 
      N=100, use_adaptive=False, tol_aaa=tol_aaa, tol_pol=1e-7, Dmax=Dmax)
                    
    all_poles.append(poles)
    all_residues.append(residues)

    if checkpointing:
      oscs_tmp = osc_scalings[:len(all_poles)]
      sorter = np.argsort(oscs_tmp)
      poles_tmp    = [all_poles[i] for i in sorter]
      residues_tmp = [all_residues[i] for i in sorter]
      oscs_tmp     = [oscs_tmp[i] for i in sorter]
      with open(fname, "wb") as file:
        pickle.dump({
          "poles":      poles_tmp, 
          "residues":   residues_tmp, 
          "osc_scaling":oscs_tmp
        }, file)
  if plotting:
    plt.figure()
    eyes(osc_scalings, all_poles, all_residues)

    ho_r = np.linspace(domain[0].real, domain[1].real, num=201)
    ho_i = np.linspace(domain[0].imag, domain[1].imag, num=201)
    hbar_omega = ho_r[:, None] + 1j*ho_i[None, :]
    plt.pcolormesh(
    eps = eps_surmof(
        hbar_omega, 
        npoles=npoles, 
        scale_osc=scale_osc, 
        scale_damping=scale_damping
    ))
    
    plt.savefig(fname.parent/(fname.stem+".png"), dpi=600)
    plt.close()
  
  return all_poles, all_residues

def eyes(ts, all_poles, all_residues):
  ts = np.abs(ts)
  cmap = mpl.cm.viridis
  norm = mpl.colors.Normalize(vmin=min(ts), vmax=max(ts))
  colors = cmap(norm(ts))
  for color, poles, residues in zip(colors, all_poles, all_residues):
      filtered_poles = poles
      plt.scatter(filtered_poles.real, filtered_poles.imag, color=color, s=np.sqrt(np.abs(residues))*100)

  plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), ax=plt.gca(), label="$\eta$")

  plt.xlabel(r"$\Re\{\hbar \tilde \omega\}$ [eV]")
  plt.ylabel(r"$\Im\{\hbar \tilde \omega\}$ [eV]")
  plt.tight_layout()

if __name__ == "__main__":
  osc_scalings = np.logspace(-3, 0, 128+1)
  domain = [1-0.5j, 2.5+0.05j]


  all_poles, all_residues = find_qnms_osc(
    osc_scalings, ds=(0, 3/4.780315, 0.3), 
    npoles=1, damping=1, domain=domain, kpar_eV=0.8,
    checkpointing=True
  )

