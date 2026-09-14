#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
from scipy.special import jv
from scipy.integrate import simpson
from concurrent.futures import ThreadPoolExecutor
import csv
import math

_int_pool = ThreadPoolExecutor(max_workers=3)

# In[2]:


def create_starting_optic(r, R, k=-1):
    z = r * r / (R + np.sqrt(R * R - (k + 1) * r * r))
    return z

def create_axiparabola(r, R, L, f0, taylor=False):
    if taylor:
        sag = r**2 / (4 * f0) * (1 - L / (2 * f0) * (r / R) ** 2)
    else:
        sag = R**2 / (4 * L) * np.log(1 + L / f0 * (r / R) ** 2)
    return sag


# def calc_Z(rho, n):
#     # pols = np.array(
#     #     [np.ones_like(rho), np.sqrt(3) * (2 * rho**2 - 1), np.sqrt(5) * (6 * rho**4 - 6 * rho**2 + 1)]
#     # )

#     zern_pols = np.array(
#         [
#             np.ones_like(rho),
#             np.sqrt(3) * (2 * rho**2 - 1),
#             np.sqrt(5) * (6 * rho**4 - 6 * rho**2 + 1),
#             np.sqrt(7) * (20 * rho**6 - 30 * rho**4 + 12 * rho**2 - 1),
#         ]
#     )
#     pols = zern_pols[:n]
#     return np.column_stack(pols)


def calc_Z(rho, j):
    # print("HELLO IVE CHANGED")

    rho = np.asarray(rho)
    Z = np.zeros((rho.size, j), dtype=float)

    for col, m in enumerate(range(0, 2 * j - 1, 2)):  # m = 0, 2, 4, ...
        res = np.zeros_like(rho, dtype=float)
        for s in range(m // 2 + 1):
            coeff = (
                ((-1) ** s)
                * math.factorial(m - s)
                / (math.factorial(s) * (math.factorial(m // 2 - s) ** 2))
            )
            res += coeff * rho ** (m - 2 * s)
        res *= math.sqrt(m + 1)  # orthonormal scaling
        Z[:, col] = res
    return Z


def calc_Z_plus(Z):
    ZTZ_inv = np.linalg.inv(Z.T @ Z)
    return ZTZ_inv @ Z.T


def create_initial_coeff(rmax, R, j):
    # print("HELLO IM RIGHT HERE")
    coeff = np.zeros(j)
    coeff[0] = rmax**2 / (2 * R) * 0.5
    coeff[1] = rmax**2 / (2 * R) * 1 / (2 * np.sqrt(3))
    # zern_coeff = rmax**2 / (2 * R) * np.array([0.5, 1 / (2 * np.sqrt(3)), 0, 0])
    # coeff = zern_coeff[:n]
    return coeff


def generate_optic(a_n, zernpol):
    return zernpol @ a_n


def DHTsetup(N, R, ell):
    # BesselZeroes*.csv lives alongside this module, not the caller's cwd --
    # shared by every directory using this Zernike parameterization.
    _csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BesselZeroes" + str(ell) + ".csv")
    Zeroes = np.genfromtxt(_csv)
    alpha = Zeroes[:N]
    kR = alpha[N - 1]
    kmax = kR / R
    r = R * alpha / alpha[N - 1]
    k = alpha / R
    T = (
        jv(ell, np.outer(alpha, alpha) / kR)
        / np.outer(abs(jv(ell + 1, alpha)), abs(jv(ell + 1, alpha)))
        / kR
    )
    A = abs(jv(ell + 1, alpha))
    T_R2K = 4.0 * np.pi * (R / kmax) * T * A[:, None] / A[None, :]
    T_K2R = (1 / np.pi) * (kmax / R) * T * A[:, None] / A[None, :]
    return r, k, T_R2K, T_K2R


# In[5]:


def propagate(fe, k, k0, z, T_R2K, T_K2R, phase=None):
    FT = T_R2K @ fe
    if phase is None:
        phase = np.exp(-1j * k**2 / (2 * k0) * z[:, np.newaxis])
    FT = FT * phase
    fT = T_K2R @ (FT.T)
    return fT


# In[6]:


# In[7]:


def mirror(phiI, k0, zm):
    return phiI * np.exp(-2j * k0 * zm)


# In[8]:


def forward_propagation(k, phi_I, T_R2K, T_K2R, zF, zm, k0, phase=None):
    phi_R0 = mirror(phi_I, k0, zm)
    fT_f = propagate(phi_R0, k, k0, zF, T_R2K, T_K2R, phase=phase)
    return fT_f


# In[10]:


def calc_intensity(ft):
    intensity = np.abs(ft) ** 2
    return intensity


def difference(obs, exp):
    diff = np.abs(np.sum(np.abs(obs) ** 2) - np.sum(np.abs(exp) ** 2)) / np.abs(
        np.sum(exp) ** 2
    )
    return diff


# In[11]:


def retrieve_bounds(x, xi, xf):
    i = len(x[x - xi < 0])
    f = len(x[x - xf <= 0])
    return i, f


# In[12]:


def w_xp(r, wR, delta):
    if delta:
        weight = np.zeros(len(r))
        weight[0] = 1
        return weight
    else:
        # return np.ones(len(r))
        return np.exp(-(r**2) / wR**2)


# In[13]:


def w_z(z, zi, zf, sg=False, order=2):
    i, f = retrieve_bounds(z, zi, zf)
    weight = np.zeros(len(z))
    if sg:
        weight[i:f] = np.exp(-(((z[i:f] - zi) / (zf - zi)) ** order))
    else:
        # weight = np.zeros(len(z))
        # weight[(f - i) // 2] = 1
        weight[i:f] = 1
    return weight


# In[14]:


def integrate(r, z, w, int1, int2, method=simpson):
    return 2 * np.pi * method(method(w * int1 * int2 * r[:, np.newaxis], z), r)


# In[15]:


def weight(r, R, z, zi, zf, delta):
    wz = w_z(z, zi, zf)
    wxp = w_xp(r, R, delta)
    w = wz * wxp[:, np.newaxis]
    return w


# In[16]:


def int_pairs(r, z, w, Ir, It):
    f_rr = _int_pool.submit(integrate, r, z, w, Ir, Ir)
    f_rt = _int_pool.submit(integrate, r, z, w, Ir, It)
    f_tt = _int_pool.submit(integrate, r, z, w, It, It)
    return f_rr.result(), f_rt.result(), f_tt.result()


# In[17]:


def FOM(Irsq, IrIt, Itsq):
    return 0.5 * (Irsq - IrIt**2 / Itsq)


# In[18]:


def calc_mu(IrIt, Itsq):
    return IrIt / Itsq


# In[19]:


def S(w, phi_r, Ir, It, IrIt, Itsq):
    mu = calc_mu(IrIt, Itsq)
    s = w * (Ir - mu * It) * np.real(np.conj(phi_r))
    # s=pr-m*pt
    return s


# In[20]:


def calc_psi_hat(k, k0, z, s_hat, phase=None):
    coeff = -1j / (2 * k0)
    if phase is None:
        phase = np.exp(-1j * np.outer(k**2, z) / (2 * k0))
    return coeff * simpson(phase * s_hat, x=z)


# In[21]:


def adjoint(r, wR, k, k0, z, zmin, zmax, phi_r, phi_t, T_R2K, delta):
    # print('pr pt')
    w = weight(r, wR, z, zmin, zmax, delta)
    # print('int pairs')
    Ir = calc_intensity(phi_r)
    It = calc_intensity(phi_t)
    _, IrIt, Itsq = int_pairs(r, z, w, Ir, It)
    # print('s')
    s = S(w, phi_r, Ir, It, IrIt, Itsq)
    s_hat = T_R2K @ s
    psi_hat = calc_psi_hat(k, k0, z, s_hat)
    return psi_hat


def matmult(Z):
    ZTZ_inv = np.linalg.inv(Z.T @ Z)
    return ZTZ_inv @ Z.T


# In[22]:
def zernike_fit(Zmat, z_m):
    a = Zmat @ z_m
    return a


def gradient(k0, phi_i, psi, a_n, Z_mir):
    z_mir = generate_optic(a_n, Z_mir)
    dfdz = 4 * k0**2 * phi_i * psi * np.exp(-2j * k0 * z_mir)
    return dfdz + np.conj(dfdz)


def step(grad, eps, a_n):
    return a_n - eps * grad


# In[28]:


def write_to_files(file_names, data):
    for file_name, lines_array in zip(file_names, data):
        with open(file_name, "a", newline="") as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=",")
            csv_writer.writerow(lines_array)


def initialize_variables_from_file(file_path):
    variables = {}

    try:
        with open(file_path, "r") as file:
            for line in file:
                # Split each line into variable name and value
                name, value = line.strip().split("=")

                # Strip whitespaces and convert value to appropriate type
                name = name.strip()
                value = eval(value.strip())

                # Store the variable in the dictionary
                variables[name] = value

        return variables
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None
