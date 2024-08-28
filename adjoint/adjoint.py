#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.special import jv
from scipy.integrate import simpson
from numpy.fft import *
from gaussian_beam import *
import csv


# In[2]:


def create_starting_optic(r, R, k=-1):
    z = r * r / (R + np.sqrt(R * R - (k + 1) * r * r))
    return z


def calc_Z(rho):
    pols = [
        np.ones_like(rho),  # 1st polynomial is constant
        np.sqrt(3) * (2 * rho**2 - 1),  # 2nd polynomial
        np.sqrt(5) * (6 * rho**4 - 6 * rho**2 + 1),  # 3rd polynomial
    ]
    return np.column_stack(pols)

def calc_Z_plus(Z):
    ZTZ_inv = np.linalg.inv(Z.T @ Z)
    return ZTZ_inv @ Z.T

def create_initial_coeff(rmax, R):
    return rmax**2 / (2 * R) * np.array([0.5, 1 / (2 * np.sqrt(3)), 0])


def generate_optic(a_n, zernpol):
    return zernpol @ a_n


def DHTarrays(N, R, ell):
    Zeroes = np.genfromtxt("BesselZeroes" + str(ell) + ".csv")
    r = R * Zeroes[:N] / Zeroes[N - 1]
    k = Zeroes[:N] / R
    return np.array([r, k])


# In[4]:


def DHTtransformMatrix(N, R, ell):
    Zeroes = np.genfromtxt("BesselZeroes" + str(ell) + ".csv")
    alpha = Zeroes[:N]
    kR = alpha[N - 1]
    kmax = kR / R
    T = (
        jv(ell, np.outer(alpha, alpha) / kR)
        / np.outer(abs(jv(ell + 1, alpha)), abs(jv(ell + 1, alpha)))
        / kR
    )
    # print(T)
    A = abs(jv(ell + 1, alpha))
    T_R2K = 4.0 * np.pi * (R / kmax) * T * A[:, None] / A[None, :]
    T_K2R = (1 / np.pi) * (kmax / R) * T * A[:, None] / A[None, :]

    return T_R2K, T_K2R


# In[5]:


def propagate(fe, k, k0, z, T_R2K, T_K2R):
    # 1) fe is the initial electric field
    # 2) Hankel transform the field through multiplying by the appropriate transform matrix
    FT = T_R2K @ fe
    # Hankel Transform
    # 3) Propagate 𝐸̃ from 𝑧 = 0 to 𝑧 = Δ𝑧 through multiplying 𝐸̃ by the diffractive phase:
    FT = FT * np.exp(-1j * k**2 / (2 * k0) * z[:, np.newaxis])
    # print(FT.shape)
    # print(T_K2R.shape)
    # 4) Inverse Hankel transform 𝐸̃ from 𝑘𝑟-space to 𝑟-space through multiplying by the inverse transform matrix
    fT = T_K2R @ (FT.T)
    return fT


# In[6]:


# In[7]:


def mirror(phiI, k0, zm):
    return phiI * np.exp(-2j * k0 * zm)


# In[8]:


def forward_propagation(w0, r, k, phi_I, T_R2K, T_K2R, zF, zm, wavelength=1e-6):
    k0 = 2 * np.pi / wavelength
    #     fT_i=prop(phi_I,k,k0,zI,T_R2K,T_K2R)
    phi_R0 = mirror(phi_I, k0, zm)

    fT_f = propagate(phi_R0, k, k0, zF, T_R2K, T_K2R)

    return fT_f


# In[10]:


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


def integrate(r, z, phi1, phi2, method=simpson):
    return (
        2
        * np.pi
        * method(method(np.real(phi1) * np.real(phi2) * r[:, np.newaxis], z), r)
    )


# In[15]:


def weighted_phi(r, R, z, zi, zf, phi_r, phi_t, delta):
    wz = w_z(z, zi, zf)
    wxp = w_xp(r, R, delta)
    pr = phi_r * wz * wxp[:, np.newaxis]
    pt = phi_t * wz * wxp[:, np.newaxis]
    return pr, pt


# In[16]:


def int_pairs(r, z, pr, pt, phi_r, phi_t):
    phi_r_sq = integrate(r, z, pr, phi_r)
    phi_r_phi_t = integrate(r, z, pr, phi_t)
    phi_t_sq = integrate(r, z, pt, phi_t)
    return phi_r_sq, phi_r_phi_t, phi_t_sq


# In[17]:


def FOM(prsq, prpt, ptsq):
    return 0.5 * (prsq - prpt**2 / ptsq)


# In[18]:


def calc_mu(prpt, ptsq):
    return prpt / ptsq


# In[19]:


def S(pr, pt, prpt, ptsq):
    mu = calc_mu(prpt, ptsq)
    s = np.real(pr) - mu * np.real(pt)
    # s=pr-m*pt
    return s


# In[20]:


def calc_psi_hat(k, k0, z, s_hat, method=simpson):
    # print('start psi hat')
    coeff = -1j / (2 * k0)
    integrand = np.exp(-1j * np.outer(k**2, z) / (2 * k0)) * s_hat
    # print('end psi hat')
    return coeff * method(integrand, z)


# In[21]:


def adjoint(r, wR, k, k0, z, zmin, zmax, phi_r, phi_t, T_R2K, delta):
    # print('pr pt')
    pR, pT = weighted_phi(r, wR, z, zmin, zmax, phi_r, phi_t, delta)
    # print('int pairs')
    prsq, prpt, ptsq = int_pairs(r, z, pR, pT, phi_r, phi_t)
    # print('s')
    s = S(pR, pT, prpt, ptsq)
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
    return dfdz


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
