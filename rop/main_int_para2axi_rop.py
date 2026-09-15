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


if len(sys.argv) != 2:
    print("Usage: python main_int_para2axi_rop.py <path to parameter file>")
    sys.exit(1)

path = sys.argv[1]
params = adj.initialize_variables_from_file(sys.argv[1])
dirname = (path.split("/")[-1])[:-4]
path = path.split("/")[0] + "/runs/int_para2axi_rop_" + dirname
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
ell = 0
f_init = params["f_init"]
conic = params["conic"]

wl = params["wl"]
k0 = 2 * np.pi / wl
fn = params["f_num"]
w0 = f_init / (2 * fn)
rmax = params["r_maxf"] * w0
nr = params["nr"]
zR = np.pi * w0**2 / wl
wR = 2 / np.pi * fn * wl
L = params["L"]

dz = params["dz"]
nzf = params["nzf"]
zF = np.linspace(f_targ - dz, f_targ + L + dz, nzf)
fl = len(zF) // 2

steps   = params["steps"]
lr      = params["lr"]
tol     = params["tol"]
delta   = params["delta"]
sg      = params.get("sg", 0)
order = params.get("order", 2)
j     = params["j"]
skip    = params["skip"]

# Reduce on plateau parameters
rop_factor    = params.get("rop_factor", 0.5)
rop_patience  = int(params.get("rop_patience", 500))
rop_threshold = params.get("rop_threshold", 1e-4)
rop_min_lr    = params.get("rop_min_lr", 1e-12)
################################################
P = params.get("P", 20)
print(w0)

r, k, T_R2K, T_K2R = adj.DHTsetup(nr, rmax, ell)
prop_phase = np.exp(-1j * k**2 / (2 * k0) * zF[:, np.newaxis])
adj_phase = prop_phase.T.copy()

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

change = 1000
i = 0
start_time = time.time()
data = []
lr_history = []
break_flag = False
wt = adj.weight(r, wR, zF, f_targ - dz, f_targ + L + dz, delta, sg, order)

# Reduce on plateau state
best_mu          = np.inf
patience_counter = 0

while change > tol and i < steps:
    print("MAIN STEP %d \t Grad:%.3e  lr=%.2e" % (i, change, lr))

    ftf_r = adj.forward_propagation(k, phi_I, T_R2K, T_K2R, zF, zm_r, k0, phase=prop_phase)

    Ir = adj.calc_intensity(ftf_r)
    Ir = Ir / maxIt
    Irsq, IrIt, Itsq = adj.int_pairs(r, zF, wt, Ir, It)
    data.append([Irsq, IrIt, Itsq])

    s     = adj.S(wt, ftf_r, Ir, It, IrIt, Itsq)
    s_hat = T_R2K @ s
    psi_hat = adj.calc_psi_hat(k, k0, zF, s_hat, phase=adj_phase)
    psi   = T_K2R @ psi_hat

    dfdz = np.real(adj.gradient(k0, phi_I, psi, a_n, Z_fit))
    dfda = Z_plus @ dfdz
    Z_plus = adj.calc_Z_plus(Z_fit)

    a_n  = adj.step(dfda, lr, a_n)
    zm_r = adj.generate_optic(a_n, Z_fit)

    if i % skip == 0:
        adj.write_to_files(
            filenames,
            [dfdz, zm_r, np.real(s[0, :]), dfda, a_n, Ir[0, :], Ir[:, fl], np.real(psi)],
        )

    mu = adj.calc_mu(IrIt, Itsq)
    lr_history.append(lr)

    # Reduce on plateau: track improvement in |μ - 1|
    mu_dist = abs(mu - 1.0)
    if best_mu - mu_dist > rop_threshold:
        best_mu = mu_dist
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= rop_patience:
            new_lr = max(lr * rop_factor, rop_min_lr)
            if new_lr < lr:
                print(f"  ROP: reducing lr {lr:.2e} -> {new_lr:.2e}")
                lr = new_lr
            patience_counter = 0

    change = np.sum(np.abs(dfda * lr))
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
