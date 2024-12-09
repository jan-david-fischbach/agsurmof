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

def sum_pair_prod(vec):
    mat = vec[None, :]*vec[:, None]
    mat[np.eye(len(mat), dtype=bool)] = 0
    return np.sum(mat)/2

def get_coupling(evs, oms):
  om_o = np.sum(evs) - np.sum(oms)
  om_a, om_b = oms
  oms = np.array([om_o, om_a, om_b])
  Pi = np.prod(oms)-np.prod(evs)
  Sigma = sum_pair_prod(oms) - sum_pair_prod(evs)
  A = (Pi-om_a*Sigma)/(om_b-om_a)
  B = Sigma-A
  return A, B, om_o

if __name__=="__main__":
    a,b = 0.5+2j, 0.8-1j
    H, *oms = example_hamiltonian(a,b)
    evs = np.linalg.eigvals(H)
    A,B = get_coupling(evs, oms)
    A,np.abs(a)**2, B,np.abs(b)**2


