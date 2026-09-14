#!/usr/bin/env python
"""
Computes and saves the optimizer's initial-guess coefficients and the exact
axiparabola target's coefficients (same physics as plot_cost_landscape.py)
as two standalone CSVs, so they can be fed into plot_cost_landscape.py (or
any other script that takes two coefficient CSVs) to interpolate between
them.

Usage:
    python save_init_and_target_coeffs.py <param_file> [out_prefix]

Writes <out_prefix>_init.csv and <out_prefix>_target.csv (out_prefix
defaults to "coeffs").
"""

import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
import adjoint_field_zernike as adj


def create_axiparabola(r, R, L, f0, taylor=False):
    if taylor:
        sag = r**2 / (4 * f0) * (1 - L / (2 * f0) * (r / R) ** 2)
    else:
        sag = R**2 / (4 * L) * np.log(1 + L / f0 * (r / R) ** 2)
    return sag


if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

param_file = sys.argv[1]
out_prefix = sys.argv[2] if len(sys.argv) >= 3 else "coeffs"

params = adj.initialize_variables_from_file(param_file)

f_targ = params["f_targ"]
ell = 0
f_init = params["f_init"]
fn = params["f_num"]
w0 = f_init / (2 * fn)
rmax = params["r_maxf"] * w0
nr = params["nr"]
L = params["L"]
num = params["n"]

r, k, T_R2K, T_K2R = adj.DHTsetup(nr, rmax, ell)

zm_t = create_axiparabola(r, w0, L, f_targ)
rho = r / rmax
Z_fit = adj.calc_Z(rho, num)
Z_plus = adj.calc_Z_plus(Z_fit)

a_target = Z_plus @ zm_t
a_init = adj.create_initial_coeff(rmax, 2 * f_init, num)

init_path = f"{out_prefix}_init.csv"
target_path = f"{out_prefix}_target.csv"
np.savetxt(init_path, a_init, delimiter=",")
np.savetxt(target_path, a_target, delimiter=",")
print("a_init:  ", a_init)
print("a_target:", a_target)
print(f"Saved to {init_path} and {target_path}")
