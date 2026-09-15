import sys
import os
import math
import numpy as np
import sys as _sys, os as _os
_sys.path.append(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "common"))
import adjoint_intensity_zernike as adj
from plot import plot_fom_mu, plot_percent_change_mu
import time
import shutil
from datetime import timedelta


def create_axiparabola(r, R, L, f0, taylor=False):
    if taylor:
        sag = r**2 / (4 * f0) * (1 - L / (2 * f0) * (r / R) ** 2)
    else:
        sag = R**2 / (4 * L) * np.log(1 + L / f0 * (r / R) ** 2)
    return sag


def cosine_lr(i_in_cycle, lr_max, lr_min, curr_T):
    """Cosine annealing lr for current position within a cycle."""
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * i_in_cycle / curr_T))


if len(sys.argv) != 2:
    print("Usage: python main_int_para2axi_sgdr.py <path to parameter file>")
    sys.exit(1)

path = sys.argv[1]
params = adj.initialize_variables_from_file(sys.argv[1])
dirname = (path.split("/")[-1])[:-4]
path = path.split("/")[0] + "/runs/int_para2axi_sgdr_" + dirname
if os.path.exists(path):
    shutil.rmtree(path)
os.makedirs(path)
subdir = ["I_OA", "I_f", "g", "mirror", "psi_S"]
for s in subdir:
    os.makedirs(path + "/" + s)

with open(path + "/time.txt", "w") as file:
    file.write(str(sys.argv[0]))

################################################
f_targ = params["f_targ"]
ell    = 0
f_init = params["f_init"]
conic  = params["conic"]

wl   = params["wl"]
k0   = 2 * np.pi / wl
fn   = params["f_num"]
w0   = f_init / (2 * fn)
rmax = params["r_maxf"] * w0
nr   = params["nr"]
wR   = 2 / np.pi * fn * wl
L    = params["L"]

dz  = params["dz"]
nzf = params["nzf"]
zF  = np.linspace(f_targ - dz, f_targ + L + dz, nzf)
fl  = len(zF) // 2

steps  = params["steps"]
lr_max = params["lr"]
lr_min = params.get("lr_min", lr_max * 1e-3)
T_0    = int(params.get("T_cycle", 200))
T_mult = params.get("T_mult", 2)
tol    = params["tol"]
delta  = params["delta"]
sg     = params.get("sg", 0)
order = params.get("order", 2)
j      = params["j"]
skip   = params["skip"]
################################################
P = params.get("P", 20)
print(w0)

r, k, T_R2K, T_K2R = adj.DHTsetup(nr, rmax, ell)
prop_phase = np.exp(-1j * k**2 / (2 * k0) * zF[:, np.newaxis])
adj_phase  = prop_phase.T.copy()

phi_I = np.exp(-((r / w0) ** 2) ** P)
zm_t  = create_axiparabola(r, w0, L, f_targ)
ftf_t = adj.forward_propagation(k, phi_I, T_R2K, T_K2R, zF, zm_t, k0)

It    = adj.calc_intensity(ftf_t)
maxIt = np.max(It)
It    = It / maxIt

a_n    = adj.create_initial_coeff(rmax, 2 * f_init, j)
rho    = r / rmax
Z_fit  = adj.calc_Z(rho, j)
Z_plus = adj.calc_Z_plus(Z_fit)
zm_r   = adj.generate_optic(a_n, Z_fit)

data_names = [
    "gradient.csv", "mirror.csv", "source.csv",
    "zernike_grad.csv", "zernike_coeff.csv",
    "int_oa.csv", "int_f.csv", "psi.csv",
]
filenames = [path + "/" + s for s in data_names]

change     = 1000
i          = 0
start_time = time.time()
data       = []
lr_history = []
break_flag = False
wt = adj.weight(r, wR, zF, f_targ - dz, f_targ + L + dz, delta, sg, order)

# SGDR cycle state
cycle_start = 0
curr_T      = T_0

while change > tol and i < steps:
    lr = cosine_lr(i - cycle_start, lr_max, lr_min, curr_T)
    print("MAIN STEP %d \t Grad:%.3e  lr=%.2e  T=%d" % (i, change, lr, curr_T))

    # Check if current cycle is done → restart with doubled cycle length
    if i - cycle_start >= curr_T:
        cycle_start = i
        curr_T      = int(curr_T * T_mult)

    lr_history.append(lr)

    # Forward propagation
    ftf_r = adj.forward_propagation(k, phi_I, T_R2K, T_K2R, zF, zm_r, k0, phase=prop_phase)

    Ir = adj.calc_intensity(ftf_r) / maxIt
    Irsq, IrIt, Itsq = adj.int_pairs(r, zF, wt, Ir, It)
    data.append([Irsq, IrIt, Itsq])

    # Adjoint
    s       = adj.S(wt, ftf_r, Ir, It, IrIt, Itsq)
    s_hat   = T_R2K @ s
    psi_hat = adj.calc_psi_hat(k, k0, zF, s_hat, phase=adj_phase)
    psi     = T_K2R @ psi_hat

    # Gradient
    dfdz = np.real(adj.gradient(k0, phi_I, psi, a_n, Z_fit))
    dfda = Z_plus @ dfdz
    Z_plus = adj.calc_Z_plus(Z_fit)

    # Plain GD update with cosine-annealed lr
    a_n  = a_n - lr * dfda
    zm_r = adj.generate_optic(a_n, Z_fit)

    if i % skip == 0:
        adj.write_to_files(
            filenames,
            [dfdz, zm_r, np.real(s[0, :]), dfda, a_n, Ir[0, :], Ir[:, fl], np.real(psi)],
        )

    change = np.sum(np.abs(lr * dfda))
    i += 1

np.savetxt(path + "/data.csv", data, delimiter=",")
np.savetxt(path + "/lr_history.csv", lr_history, delimiter=",")
final_time = time.time()

if i < steps and not break_flag:
    output = "Algorithm converged in %d steps \nTotal Time:" % i + str(timedelta(seconds=final_time - start_time))
else:
    output = "Algorithm failed to converge in %d steps \nTotal Time:" % i + str(timedelta(seconds=final_time - start_time))

print(output)
rr, rt, tt = np.genfromtxt(path + "/data.csv", delimiter=",", unpack=True)
plot_fom_mu(path, adj.FOM(rr, rt, tt), adj.calc_mu(rt, tt))
plot_percent_change_mu(path, adj.calc_mu(rt, tt))

with open(path + "/time.txt", "a") as file:
    file.write("\n" + output)

shutil.move(sys.argv[1], path)
print("Finished running main function")
