#!/usr/bin/env python
# -*- coding: utf-8 -*-
# zernike.py
"""
A module defining the zernike polynomials and associated functions.

Running this file as a script will output a graph of the first 15 zernike
polynomials on the unit disk.

https://en.wikipedia.org/wiki/Zernike_polynomials
http://mathworld.wolfram.com/ZernikePolynomial.html

Copyright (c) 2016, David Hoffman
"""

from typing import Tuple

import numpy as np
from scipy.special import eval_jacobi
from matplotlib.colors import LinearSegmentedColormap

cm_data = [
    [0.2081, 0.1663, 0.5292],
    [0.2116238095, 0.1897809524, 0.5776761905],
    [0.212252381, 0.2137714286, 0.6269714286],
    [0.2081, 0.2386, 0.6770857143],
    [0.1959047619, 0.2644571429, 0.7279],
    [0.1707285714, 0.2919380952, 0.779247619],
    [0.1252714286, 0.3242428571, 0.8302714286],
    [0.0591333333, 0.3598333333, 0.8683333333],
    [0.0116952381, 0.3875095238, 0.8819571429],
    [0.0059571429, 0.4086142857, 0.8828428571],
    [0.0165142857, 0.4266, 0.8786333333],
    [0.032852381, 0.4430428571, 0.8719571429],
    [0.0498142857, 0.4585714286, 0.8640571429],
    [0.0629333333, 0.4736904762, 0.8554380952],
    [0.0722666667, 0.4886666667, 0.8467],
    [0.0779428571, 0.5039857143, 0.8383714286],
    [0.079347619, 0.5200238095, 0.8311809524],
    [0.0749428571, 0.5375428571, 0.8262714286],
    [0.0640571429, 0.5569857143, 0.8239571429],
    [0.0487714286, 0.5772238095, 0.8228285714],
    [0.0343428571, 0.5965809524, 0.819852381],
    [0.0265, 0.6137, 0.8135],
    [0.0238904762, 0.6286619048, 0.8037619048],
    [0.0230904762, 0.6417857143, 0.7912666667],
    [0.0227714286, 0.6534857143, 0.7767571429],
    [0.0266619048, 0.6641952381, 0.7607190476],
    [0.0383714286, 0.6742714286, 0.743552381],
    [0.0589714286, 0.6837571429, 0.7253857143],
    [0.0843, 0.6928333333, 0.7061666667],
    [0.1132952381, 0.7015, 0.6858571429],
    [0.1452714286, 0.7097571429, 0.6646285714],
    [0.1801333333, 0.7176571429, 0.6424333333],
    [0.2178285714, 0.7250428571, 0.6192619048],
    [0.2586428571, 0.7317142857, 0.5954285714],
    [0.3021714286, 0.7376047619, 0.5711857143],
    [0.3481666667, 0.7424333333, 0.5472666667],
    [0.3952571429, 0.7459, 0.5244428571],
    [0.4420095238, 0.7480809524, 0.5033142857],
    [0.4871238095, 0.7490619048, 0.4839761905],
    [0.5300285714, 0.7491142857, 0.4661142857],
    [0.5708571429, 0.7485190476, 0.4493904762],
    [0.609852381, 0.7473142857, 0.4336857143],
    [0.6473, 0.7456, 0.4188],
    [0.6834190476, 0.7434761905, 0.4044333333],
    [0.7184095238, 0.7411333333, 0.3904761905],
    [0.7524857143, 0.7384, 0.3768142857],
    [0.7858428571, 0.7355666667, 0.3632714286],
    [0.8185047619, 0.7327333333, 0.3497904762],
    [0.8506571429, 0.7299, 0.3360285714],
    [0.8824333333, 0.7274333333, 0.3217],
    [0.9139333333, 0.7257857143, 0.3062761905],
    [0.9449571429, 0.7261142857, 0.2886428571],
    [0.9738952381, 0.7313952381, 0.266647619],
    [0.9937714286, 0.7454571429, 0.240347619],
    [0.9990428571, 0.7653142857, 0.2164142857],
    [0.9955333333, 0.7860571429, 0.196652381],
    [0.988, 0.8066, 0.1793666667],
    [0.9788571429, 0.8271428571, 0.1633142857],
    [0.9697, 0.8481380952, 0.147452381],
    [0.9625857143, 0.8705142857, 0.1309],
    [0.9588714286, 0.8949, 0.1132428571],
    [0.9598238095, 0.9218333333, 0.0948380952],
    [0.9661, 0.9514428571, 0.0755333333],
    [0.9763, 0.9831, 0.0538],
]

parula_map = LinearSegmentedColormap.from_list("parula", cm_data)
# For use of "viscm view"
test_cm = parula_map
# classical names for the Noll indices
# https://en.wikipedia.org/wiki/Zernike_polynomials
degrees2name = {
    (0, 0): "piston",
    # 1st order
    (1, -1): "vertical tilt",
    (1, 1): "horizontal tilt",
    # 2nd order
    (2, -2): "oblique astigmatism",
    (2, 0): "defocus",
    (2, 2): "vertical astigmatism",
    # 3rd order
    (3, -3): "vertical trefoil",
    (3, -1): "vertical coma",
    (3, 1): "horizontal coma",
    (3, 3): "oblique trefoil",
    # 4th order
    (4, -4): "oblique quadrafoil",  # sometimes called tetrafoil
    (4, -2): "oblique secondary astigmatism",
    (4, 0): "primary spherical",
    (4, 2): "vertical secondary astigmatism",
    (4, 4): "vertical quadrafoil",
    # 5th order
    # (5, -5): "vertical pentafoil",
    (5, -3): "vertical secondary trefoil",
    (5, -1): "vertical secondary coma",
    (5, 1): "horizontal seconday coma",
    (5, 3): "horizontal secondary trefoil",
    # (5, 5): "horizontal pentafoil",
    # 6th order
    # (6, -6): "vertical hexafoil",
    (6, -4): "oblique secondary quadrafoil",
    (6, -2): "oblique tertiary astigmatism",
    (6, 0): "secondary spherical",
    (6, 2): "vertical tertiary astigmatism",
    (6, 4): "vertical secondary quadrafoil",
    # (6, -6): "horizontal hexafoil",
    # 7th order
    (7, -3): "vertical tertiary trefoil",
    (7, -1): "vertical tertiary coma",
    (7, 1): "horizontal tertiary coma",
    (7, 3): "horizontal tertiary trefoil",
    # 8th order
    (8, -2): "oblique quaternary astigmatism",
    (8, 0): "tertiary spherical",
    (8, 2): "vertical quaternary astigmatism",
    # 9th order
    (9, -3): "vertical quaternary trefoil",
    (9, -1): "vertical quaternary coma",
    (9, 1): "horizontal quaternary coma",
    (9, 3): "horizontal quaternary trefoil",
    # 10th order
    (10, 0): "quaternary spherical",
}

name2degrees = {v: k for k, v in degrees2name.items()}


assert len(name2degrees) == len(degrees2name)


def cart2pol(y, x):
    """Convert from cartesian to polar coordinates."""
    theta = np.arctan2(y, x)
    rho = np.hypot(y, x)
    return rho, theta


def _ingest_degrees(n, m):
    """Convert inputs to arrays and do type and validity checking."""
    n, m = np.asarray(n), np.asarray(m)

    # check inputs
    if np.any((n < abs(m)) | ((n - m) % 2 == 1)):
        raise ValueError(f"Invalid combination of ({n}, {m}) in Noll indexing")

    if not np.issubdtype(n.dtype, np.signedinteger):
        raise ValueError("Radial degree is not integer, input = {n}")

    if not np.issubdtype(m.dtype, np.signedinteger):
        raise ValueError("Azimuthal degree is not integer, input = {m}")

    return n, m


def degrees2osa(n: int, m: int) -> int:
    """Convert Zernike polynomial radial degree (n) and azimuthal degree (m) to OSA/ANSI sequential indices.

    NOTE: inputs are vectorized and outputs will be ndarrays

    Source: "Standards for Reporting the Optical Aberrations of Eyes", Journal of Refractive Surgery Volume 18 September/October 2002
    Converted from https://github.com/rdoelman/ZernikePolynomials.jl/blob/2825846679607f7bf335fdb9edd3b7145d65082b/src/ZernikePolynomials.jl
    """
    n, m = _ingest_degrees(n, m)

    return ((1 / 2) * (n * (n + 2) + m)).astype(int)


def degrees2fringe(n: int, m: int) -> int:
    """Convert Zernike polynomial radial degree (n) and azimuthal degree (m) to Fringe sequential indices.

    NOTE: inputs are vectorized and outputs will be ndarrays
    """
    n, m = _ingest_degrees(n, m)

    return ((1 + (n + abs(m)) / 2) ** 2 - 2 * abs(m) + (1 - np.sign(m)) / 2).astype(int)


def degrees2noll(n: int, m: int) -> int:
    """Convert Zernike polynomial radial degree (n) and azimuthal degree (m) to Noll sequential indices.

    NOTE: inputs are vectorized and outputs will be ndarrays

    Source: "Standards for Reporting the Optical Aberrations of Eyes", Journal of Refractive Surgery Volume 18 September/October 2002
    Converted from https://github.com/rdoelman/ZernikePolynomials.jl/blob/2825846679607f7bf335fdb9edd3b7145d65082b/src/ZernikePolynomials.jl
    """
    n, m = _ingest_degrees(n, m)

    p = np.full_like(m, -1)
    n_mod_4 = n % 4
    p[(m > 0) & (n_mod_4 < 2)] = 0
    p[(m < 0) & (n_mod_4 > 1)] = 0
    p[(m >= 0) & (n_mod_4 > 1)] = 1
    p[(m <= 0) & (n_mod_4 < 2)] = 1

    if np.any(p < 0):
        raise RuntimeError
    return (n * (n + 1) / 2 + abs(m) + p).astype(int)


def _ingest_index(j):
    """Convert inputs to arrays and do type and validity checking."""
    j = np.asarray(j)

    # check inputs
    if not (j > 0).all():
        raise ValueError("Invalid Noll index")
    if not np.issubdtype(j.dtype, np.signedinteger):
        raise ValueError("Index is not integer, input = {j}")
    return j


def osa2degrees(j: int) -> Tuple[int, int]:
    """Convert the sequential OSA/ANSI stardard index number j to the integer pair (n, m) that defines the Zernike polynomial Z_n^m(ρ, θ).

    NOTE: inputs are vectorized and outputs will be ndarrays

    Source: "Standards for Reporting the Optical Aberrations of Eyes", Journal of Refractive Surgery Volume 18 September/October 2002
    https://github.com/rdoelman/ZernikePolynomials.jl/blob/2825846679607f7bf335fdb9edd3b7145d65082b/src/ZernikePolynomials.jl
    """
    j = _ingest_index(j)

    n = (np.ceil((-3 + np.sqrt(9 + 8 * j)) / 2)).astype(int)
    m = 2 * j - n * (n + 2)
    return n.astype(int), m.astype(int)


def noll2degrees(j: int) -> Tuple[int, int]:
    """Convert the Noll index number j to the integer pair (n, m) that defines the Zernike polynomial Z_n^m(ρ, θ).

    NOTE: inputs are vectorized and outputs will be ndarrays

    Source: (https://en.wikipedia.org/wiki/Zernike_polynomials)
    https://github.com/rdoelman/ZernikePolynomials.jl/blob/2825846679607f7bf335fdb9edd3b7145d65082b/src/ZernikePolynomials.jl
    """
    j = _ingest_index(j)

    n = (np.ceil((-3 + np.sqrt(1 + 8 * j)) / 2)).astype(int)
    jr = j - (n * (n + 1) / 2).astype(int)

    # if (n % 4) < 2:
    #     m1 = jr
    #     m2 = -(jr - 1)
    #     if (n - m1) % 2 == 0:
    #         m = m1
    #     else:
    #         m = m2
    # else:  # mod(n,4) ∈ (2,3)
    #     m1 = jr - 1
    #     m2 = -(jr)
    #     if (n - m1) % 2 == 0:
    #         m = m1
    #     else:
    #         m = m2

    # below is the vectorization version of the above.

    m1 = np.zeros_like(jr)
    m2 = np.zeros_like(jr)
    m = np.zeros_like(jr)

    n_mod_4 = n % 4

    idx0 = n_mod_4 < 2
    m1[idx0] = jr[idx0]
    m2[idx0] = -(jr[idx0] - 1)

    m1[~idx0] = jr[~idx0] - 1
    m2[~idx0] = -jr[~idx0]

    idx1 = (n - m1) % 2 == 0
    m[idx1] = m1[idx1]
    m[~idx1] = m2[~idx1]

    return n, m


def fringe2degrees(j: int):
    """Convert the Fringe index number j to the integer pair (n, m) that defines the Zernike polynomial Z_n^m(ρ, θ).

    NOTE: inputs are vectorized and outputs will be ndarrays
    """
    raise NotImplementedError


# pre-computed mappings
noll2name = {degrees2noll(n, m): name for (n, m), name in degrees2name.items()}
# sort
noll2name = dict(sorted(noll2name.items()))
# reverse
name2noll = {v: k for k, v in noll2name.items()}

osa2name = {degrees2osa(n, m): name for (n, m), name in degrees2name.items()}
# sort
osa2name = dict(sorted(osa2name.items()))
# reverse
name2osa = {v: k for k, v in osa2name.items()}

fringe2name = {degrees2fringe(n, m): name for (n, m), name in degrees2name.items()}
# sort
fringe2name = dict(sorted(fringe2name.items()))
# reverse
name2fringe = {v: k for k, v in fringe2name.items()}


def zernike(r: float, theta: float, n: int, m: int, norm: bool = True) -> float:
    """Calculate the Zernike polynomial on the unit disk for the requested orders.

    Parameters
    ----------
    r : ndarray
    theta : ndarray

    Args
    ----
    Noll : numeric or numeric sequence
        Noll's Indices to generate
    (n, m) : tuple of numerics or numeric sequences
        Radial and azimuthal degrees
    n : see above
    m : see above

    Kwargs
    ------
    norm : bool (default False)
        Do you want the output normed?

    Returns
    -------
    zernike : ndarray
        The zernike polynomials corresponding to Noll or (n, m) whichever are
        provided

    Example
    -------
    >>> x = np.linspace(-1, 1, 512)
    >>> xx, yy = np.meshgrid(x, x)
    >>> r, theta = cart2pol(yy, xx)
    >>> zern = zernike(r, theta, 4, 0)  # generates the defocus zernike polynomial
    """
    n, m = np.asarray(n), np.asarray(m)
    if n.ndim > 1:
        raise ValueError("Radial degree has the wrong shape")
    if m.ndim > 1:
        raise ValueError("Azimuthal degree has the wrong shape")
    if n.shape != m.shape:
        raise ValueError("Radial and Azimuthal degrees have different shapes")

    # make sure r and theta are arrays
    r = np.asarray(r, dtype=float)
    theta = np.asarray(theta, dtype=float)

    # make sure that r is always greater than 0
    if not (r >= 0).all():
        raise ValueError("r must always be greater or equal to 0")
    if r.ndim > 2:
        raise ValueError("Input rho and theta cannot have more than two dimensions")

    # make sure that n and m are iterable
    n, m = n.ravel(), m.ravel()

    # make sure that n is always greater or equal to m
    if not (n >= abs(m)).all():
        raise ValueError("n must always be greater or equal to m")

    # return column of zernike polynomials
    return np.array(
        [_zernike(r, theta, nn, mm, norm) for nn, mm in zip(n, m)]
    ).squeeze()


def _radial_zernike(r, n, m):
    """Radial part of the zernike polynomial.

    Formula from http://mathworld.wolfram.com/ZernikePolynomial.html
    """
    rad_zern = np.zeros_like(r)
    # zernike polynomials are only valid for r <= 1
    valid_points = r <= 1.0
    if m == 0 and n == 0:
        rad_zern[valid_points] = 1
        return rad_zern
    rprime = r[valid_points]
    # for the radial part m is always positive
    m = abs(m)
    # calculate the coefs
    coef1 = (n + m) // 2
    coef2 = (n - m) // 2
    jacobi = eval_jacobi(coef2, m, 0, 1 - 2 * rprime**2)
    rad_zern[valid_points] = (-1) ** coef2 * rprime**m * jacobi
    return rad_zern


def _zernike(r, theta, n, m, norm=True):
    """Calculate the full zernike polynomial."""
    # remember if m is negative
    mneg = m < 0
    # going forward m is positive (Radial zernikes are only defined for
    # positive m)
    m = abs(m)
    # if m and n aren't seperated by multiple of two then return zeros
    if (n - m) % 2:
        return np.zeros_like(r)
    zern = _radial_zernike(r, n, m)
    if mneg:
        # odd zernike
        zern *= np.sin(m * theta)
    else:
        # even zernike
        zern *= np.cos(m * theta)

    # calculate the normalization factor
    if norm:
        # https://www.gatinel.com/en/recherche-formation/wavefront-sensing/zernike-polynomials/
        if m == 0:
            # m is zero
            norm = np.sqrt(n + 1)
        else:
            # m not zero
            norm = np.sqrt(2 * (n + 1))
        zern *= norm
    return zern


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    # make coordinates
    x = np.linspace(-1, 1, 513)
    xx, yy = np.meshgrid(x, x)  # xy indexing is default
    r, theta = cart2pol(yy, xx)
    # set up plot
    fig, axs = plt.subplots(6, 6, figsize=(20, 12))
    # fill out plot
    for ax, ((n, m), v) in zip(axs.ravel(), degrees2name.items()):
        zern = zernike(r, theta, n, m, norm=False)
        print("zern")
        ax.imshow(
            np.ma.array(zern, mask=r > 1),
            vmin=-1,
            vmax=1,
            cmap=parula_map,
            interpolation="bicubic",
        )
        ax.set_title(v + r", $Z_{{{}}}^{{{}}}$".format(n, m))
        ax.axis("off")
    fig.tight_layout()
    plt.savefig("zernike.png")

print("done")
