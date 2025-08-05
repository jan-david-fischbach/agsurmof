# %% [markdown]
# # Given the eigenfrequencies, what are the coupling parameters?
# 
# Let's start with a System consisting of the optical bath and two material resonances described by the Hamiltonian in matrix form:
# $$
# \mathbf{H} = \begin{bmatrix}
# \tilde{\omega}_0 & a & b\\
# a^* & \tilde{\omega}_a & 0\\
# b^* & 0 & \tilde{\omega}_b\end{bmatrix}
# $$
# 
# where $a$ and $b$ are the coupling coefficients to material resonance a and b respectively.

# %%
import numpy as np
def example_hamiltonian(a=0.8, b=0.5, om_o=2, om_a=5, om_b=2):
  H = [
    [om_o,       a,    b   ],
    [np.conj(a), om_a, 0   ],
    [np.conj(b), 0,    om_b]
  ]
  return H, om_a, om_b

def example_hamiltonian_4material_modes(a=0.8, b=0.5, om_o=2, om_a=5, om_b=2):
  H = [
    [om_o,       a,    b,    2*a,    2*b],
    [np.conj(a), om_a, 0,    0,      0],
    [np.conj(b), 0,    om_b, 0,      0],
    [2*np.conj(a), 0,  0,    om_a*3, 0],
    [2*np.conj(b), 0,  0,    0,      om_b*4],
  ]
  return H, om_a, om_b, 3*om_a, 4*om_b

def sum_pair_prod(vec):
    mat = vec[None, :]*vec[:, None]
    mat[np.eye(len(mat), dtype=bool)] = 0
    return np.sum(mat)/2

def get_coupling(evs, oms):
  """For two material poles"""
  om_o = np.sum(evs) - np.sum(oms)
  om_a, om_b = oms
  oms = np.array([om_o, om_a, om_b])
  Pi = np.prod(oms)-np.prod(evs)
  Sigma = sum_pair_prod(oms) - sum_pair_prod(evs)
  A = (Pi-om_a*Sigma)/(om_b-om_a)
  B = Sigma-A
  return A, B, om_o

def P_lam(evs, oms):
  evs = evs[..., :, None]
  oms = oms[..., None, :]
  return np.prod(oms-evs, axis=-1)

def mat_A(evs, oms):
  P = P_lam(evs, oms)[:, None]
  evs = evs[..., :, None]
  oms = oms[..., None, :]

  fac = -1/(oms-evs)
  fac = np.insert(fac, 0, 1, -1)
  return P*fac

def vec_b(evs, oms):
  return P_lam(evs, oms)*evs
  
def solve_inv_eig(evs, oms):
  """
  For an arbitrary number of material poles

  """
  evs=np.atleast_1d(evs)
  oms=np.atleast_1d(oms)
  if evs.shape[-1]!=oms.shape[-1]+1:
    raise ValueError("Number of eigenvalues has to be exactly one bigger than number of material resonances")
  return np.linalg.solve(mat_A(evs, oms), vec_b(evs, oms))

def test_fwd_eig(om_o, oms, coupling):
  if np.any(np.isnan(coupling)):
    return np.ones(len(oms)+1)*np.nan
  oms = np.diag(oms)
  coupling = np.atleast_2d(coupling)
  om_o = np.atleast_2d(om_o)
  to_block = [
    [om_o,         coupling],
    [np.conj(coupling).T, oms],
  ]
  H = np.block(to_block)
  evs = np.linalg.eigvals(H)
  return evs


if __name__=="__main__":
    a,b = 1j, 2
    # H, *oms = example_hamiltonian(a,b)
    # evs = np.linalg.eigvals(H)
    # A,B, om_o = get_coupling(evs, oms)
    # print(f"{A=},{np.abs(a)**2=}\n{B=},{np.abs(b)**2=}")

    # om_o, A,B = solve_inv_eig(evs, oms)
    # print(f"{A=},{np.abs(a)**2=}\n{B=},{np.abs(b)**2=}")
    # print(f"{om_o=}")

    H, *oms = example_hamiltonian_4material_modes(a,b)

    print(f"{H=}")
    evs = np.linalg.eigvals(H)

    om_o, A,B,C,D = solve_inv_eig(evs, oms)
    print(f"{A=}")
    print(f"{B=}")
    print(f"{C=}")
    print(f"{D=}")


