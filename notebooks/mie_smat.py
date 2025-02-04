#####
# Core Multishell Mie coefficients by S-matrix
#
# References:
# [1]: PhD thesis Dominik Beutel
# 
#####
import treams.special as ts
import treams
from functools import partial
import numpy as np

def bessel_hankel(n, bessel):
    def inner(x):
        return ts.spherical_jn(n, x) if bessel else ts.spherical_hankel1(n, x)
    return inner

def bessel_hankel_d(n, bessel):
    def inner(x):
        return ts.spherical_jn_d(n, x) if bessel else ts.spherical_hankel1_d(n, x)
    return inner

def chi(l, x, bessel: bool) -> complex:
    r"""The bessel/hankel Chi elements (used in second row of 2.40 and 2.41)
    $\Chi_l^{(n)}(x) = \frac{(xz_l^{(n)})^\prime}{x}$
    Args:
        l (int): multipole degree (angular momentum)
        x (complex): size parameter
        bessel (bool): use bessel function (else hankelöfunction of first type)
    """

    z   = bessel_hankel(l, bessel)
    z_d = bessel_hankel_d(l, bessel)
    return (z(x) + x*z_d(x))/x

def psi(l, xm, xp, bessel=True):
    """The bessel/hankel Psi matrix as given in eq. 2.40 of [1]

    Args:
        l (int): multipole degree (angular momentum)
        xm (complex): size parameter for - helicity 
        xp (complex): size parameter for + helicity
        bessel (bool): use bessel function (else hankelöfunction of first type)
    """
    z = bessel_hankel(l, bessel)
    return np.array([
        [-z(xm), z(xp)],
        [chi(l, xm, bessel), chi(l, xp, bessel)]
    ])

def xi(l, xm, xp, Zj, bessel=True):
    """The bessel/hankel Xi matrix as given in eq. 2.41 of [1]

    Args:
        l (int): multipole degree (angular momentum)
        xm (complex): size parameter for - helicity 
        xp (complex): size parameter for + helicity
        Zj (complex): Impedance in layer j
        bessel (bool): use bessel function (else hankelöfunction of first type)
    """
    z = bessel_hankel(l, bessel)
    return 1/Zj* np.array([
        [z(xm), z(xp)],
        [-chi(l, xm, bessel), chi(l, xp, bessel)]
    ])

def S_mat(l, x, z):
    """The S-matrix of one layer including its outer inteface

    Args:
        l (int): multipole degree (angular momentum)
        x (2x2 array[complex]): size parameter 
            for different sides of the interface (first index) 
            and different helicities (second index)
        z (2x1 array[complex]): impedance for different sides of the interface
    """
    A = psi(l, *x[1], True)
    B = psi(l, *x[1], False)
    C =  xi(l, *x[1], z[1], True)
    D =  xi(l, *x[1], z[1], False)

    E = psi(l, *x[0], True)
    F = psi(l, *x[0], False)
    G =  xi(l, *x[0], z[0], True)
    H =  xi(l, *x[0], z[0], False)

    right = np.block([
        [A, -F],
        [C, -H]
    ])

    left = np.block([
        [E, -B],
        [G, -D]
    ])

    S = np.linalg.solve(left, right)
    return S

def stack_S(inner, outer):
    """Stack two S matrices (copied from treams)

    Args:
        inner (4x4 S matrix): S matrix of inner layers
        outer (4x4 S matrix): S matrix of outer layers

    Returns:
        4x4 S matrix: stacked S matrix
    """
    dim = 2 # polarizations
    inner = split_blocks(inner, dim)
    outer = split_blocks(outer, dim)

    try:
        snew = [[None, None], [None, None]]
        s_tmp = np.linalg.solve(np.eye(dim) - outer[0, 1] @ inner[1][0], outer[0, 0])
        snew[0][0] = inner[0][0] @ s_tmp
        snew[1][0] = outer[1, 0] + outer[1, 1] @ inner[1][0] @ s_tmp
        s_tmp = np.linalg.solve(np.eye(dim) - inner[1][0] @ outer[0, 1], inner[1][1])
        snew[1][1] = outer[1, 1] @ s_tmp
        snew[0][1] = inner[0][1] + inner[0][0] @ outer[0, 1] @ s_tmp
        return np.block(snew)
    except np.linalg.LinAlgError as err:
        print(f"warning: encounterd linalg error: {err}")
        return np.eye(4)

def split_blocks(m, idx=2):

    return np.array([
        [m[:idx, :idx], m[:idx, idx:]], 
        [m[idx:, :idx], m[idx:, idx:]]
    ])

def mie(l, x0, epsilon, mu, kappa):
    """The S-matrix of one layer including its outer inteface

    Args:
        l (int): multipole degree (angular momentum)
        x0 (list[complex]): size parameters $k_0r$ of the interfaces 
            (one entry shorter than the material parameters)
        epsilon (list[complex]): realtive permittivity of the layers
        mu (list[complex]): realtive permeability of the layers
        kappa (list[complex]): chirality parameter of the layers
    """
    z_outer  = np.sqrt(mu[0]/epsilon[0])
    nr_outer = np.sqrt(mu[0]*epsilon[0])
    for i, x_i in enumerate(x0):
        z_inner  = z_outer
        nr_inner = nr_outer

        z_outer  = np.sqrt(mu[i+1]/epsilon[i+1])
        nr_outer = np.sqrt(mu[i+1]*epsilon[i+1])

        x = x_i*np.array([
            [nr_inner - kappa[i],   nr_inner + kappa[i]],
            [nr_outer - kappa[i+1], nr_outer - kappa[i+1]]
        ])

        if i == 0:
            S = S_mat(l, x, [z_inner, z_outer])
        else:
            S = stack_S(S, S_mat(l, x, [z_inner, z_outer]))
        
    
    [[S11, S12], [S21, S22]] = split_blocks(S)
    return S21
    #return np.linalg.inv(S12 - S11@np.linalg.solve(S21, S22))

if __name__ == "__main__":
    k = 1
    l = 1

    out = mie(l, k * np.array([2,3]), [4,2,1], [1,1,1], [0,0,0])
    out_treams = treams.coeffs.mie(l, k * np.array([2,3]), [4,2,1], [1,1,1], [0,0,0])
    print(f"{out=}")
    print(f"{out_treams=}")