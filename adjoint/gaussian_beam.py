#!/usr/bin/env python
# coding: utf-8

# In[8]:
import numpy as np
from scipy.optimize import curve_fit
import math


def w(z, zR, w0):
    return w0 * np.sqrt((1 + (z / zR) ** 2))


# In[9]:


def R_inv(z, zR):
    return z / (z**2 + zR**2)


# In[10]:


# def gauss(r, z, zR, k0, w0, lens=False):
#     amp = (w0 / w(z, zR, w0)) * np.exp(-((r / w(z, zR, w0)) ** 2))
#     longitudinal = np.exp(1j * (k0 * z - np.arctan(z / zR)))
#     # radial=1 if z==0 else np.exp(1j*k0*r**2*R_inv(z,zR)/2)
#     radial = np.exp(1j * k0 * r**2 * R_inv(z, zR) / 2)
#     return amp * longitudinal * radial
def gauss(r, z, zR, k0, w0, lens=False):
    r = r[:, np.newaxis]  # shape (len(r), 1)
    z = z[np.newaxis, :]  # shape (1, len(z))

    # Compute the amplitude
    amp = (w0 / w(z, zR, w0)) * np.exp(-((r / w(z, zR, w0)) ** 2))

    # Compute the longitudinal component
    longitudinal = np.exp(1j * (k0 * z - np.arctan(z / zR)))

    # Compute the radial component
    radial = np.exp(1j * k0 * r**2 * R_inv(z, zR) / 2)

    return amp * longitudinal * radial


# In[7]:


def g_beam(r, a, w0sq):
    return a * np.exp(-(r**2) / w0sq)


# In[8]:


def find_waist(r, g, p0):
    popt, pcov = curve_fit(g_beam, r, g, p0)

    return popt, np.sqrt(np.diag(pcov))[1]


# In[9]:


def lens_R(f, z, w0sq, wavelength=1e-6):
    z01 = np.pi * w0sq / wavelength

    numerator = (1 - z / f) ** 2 + (z / z01) ** 2
    denominator = -1 / f + z * (1 / (f**2) + 1 / (z01**2))
    return numerator / denominator


# In[10]:


def lens_w(f, z, w0sq, wavelength=1e-6):
    z01 = np.pi * w0sq / wavelength

    numerator = (1 - z / f) ** 2 + (z / z01) ** 2
    denominator = 1 / z01
    return np.sqrt(wavelength / np.pi * numerator / denominator)


# In[11]:


def zM(f, wsq, wavelength=1e-6):
    z01 = np.pi * wsq / wavelength

    return f / (1 + (f / z01) ** 2)


# In[12]:


def lens_amp(w0, f, z, r):
    return (w0 / lens_w(f, z, w0**2)) * np.exp(
        -((r[:, np.newaxis] / lens_w(f, z, w0**2)) ** 2)
    )


def lens_gauss(r, f, z, zR, k0, w0, w0sq):
    amp = lens_amp(w0, f, z, r)
    longitudinal = np.exp(-1j * (k0 * z - np.arctan(z / zR)))
    radial = np.exp(-1j * k0 * r**2 / (2 * lens_R(f, z, w0sq)))
    # radial=1 if z==0 else np.exp(1j*k0*r**2/(2*lens_R(f,z,w0sq)))

    # radial=1
    return amp * longitudinal * radial
