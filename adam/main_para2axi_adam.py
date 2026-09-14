import sys
import os
import math
import numpy as np
import sys as _sys, os as _os
_sys.path.append(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "common"))
import adjoint_field_zernike as adj
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
    print("Usage: python fom.py <path to parameter file>")
    sys.exit(1)

path = sys.argv[1]
params = adj.initialize_variables_from_file(sys.argv[1])
dirname = (path.split("/")[-1])[:-4]
path = path.split("/")[0] + "/runs/para2axi_adam_" + dirname
if os.path.exists(path):
    shutil.rmtree(path)
os.makedirs(path)
subdir = ["phr_f", "pr_oa", "I_OA", "I_f", "g", "psi_S", "mirror"]
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
zi = f_targ - dz
zf = f_targ + L + dz
zF = np.linspace(zi, zf, nzf)
fl = len(zF) // 2

OoM = math.floor(math.log(w0**2, 10))
w_guess = 10**OoM
steps = params["steps"]
lr = params["lr"]
tol = params["tol"]
delta = params["delta"]
j = params["j"]
skip = params["skip"]
stop = params["stop"]

beta1 = params.get("beta1", 0.9)
beta2 = params.get("beta2", 0.999)
eps_adam = params.get("eps_adam", 1e-8)
################################################
print(w0)

r, k, T_R2K, T_K2R = adj.DHTsetup(nr, rmax, ell)
prop_phase = np.exp(-1j * k**2 / (2 * k0) * zF[:, np.newaxis])
adj_phase = prop_phase.T.copy()

phi_I = np.exp(-((r / w0) ** 40))
zm_t = create_axiparabola(r, w0, L, f_targ)
ftf_t = adj.forward_propagation(k, phi_I, T_R2K, T_K2R, zF, zm_t, k0)
It = adj.calc_intensity(ftf_t)

a_n = adj.create_initial_coeff(rmax, 2 * f_init, j)
print(a_n)
rho = r / rmax
Z_fit = adj.calc_Z(rho, j)
Z_plus = adj.calc_Z_plus(Z_fit)
zm_r = adj.generate_optic(a_n, Z_fit)

data_names = [
    "phi_r_OA.csv",
    "phi_r_focus.csv",
    "gradient.csv",
    "mirror.csv",
    "source.csv",
    "zernike_grad.csv",
    "zernike_coeff.csv",
    "int_oa.csv",
    "int_f.csv",
    "psi.csv",
]
filenames = [path + "/" + s for s in data_names]

m = np.zeros_like(a_n)
v = np.zeros_like(a_n)

change = 1000
i = 0
start_time = time.time()
data = []
break_flag = False

while change > tol and i < steps:
    if i % 1 == 0:
        print("MAIN STEP %d \t Grad:%.3e" % (i, change))

    ftf_r = adj.forward_propagation(k, phi_I, T_R2K, T_K2R, zF, zm_r, k0, phase=prop_phase)
    Ir = adj.calc_intensity(ftf_r)

    pR, pT = adj.weighted_phi(r, wR, zF, f_targ - dz, f_targ + L + dz, ftf_r, ftf_t, delta)
    prsq, prpt, ptsq = adj.int_pairs(r, zF, pR, pT, ftf_r, ftf_t)
    data.append([prsq, prpt, ptsq])

    s = adj.S(pR, pT, prpt, ptsq)
    s_hat = T_R2K @ s
    psi_hat = adj.calc_psi_hat(k, k0, zF, s_hat, phase=adj_phase)
    psi = T_K2R @ psi_hat

    dfdz = np.real(adj.gradient(k0, phi_I, psi, a_n, Z_fit))
    dfda = Z_plus @ dfdz
    Z_plus = adj.calc_Z_plus(Z_fit)

    t = i + 1
    m = beta1 * m + (1 - beta1) * dfda
    v = beta2 * v + (1 - beta2) * dfda**2
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    adam_step = lr * m_hat / (np.sqrt(v_hat) + eps_adam)
    a_n = a_n - adam_step
    zm_r = adj.generate_optic(a_n, Z_fit)

    if i % skip == 0:
        phi_r_oa = np.real(ftf_r[0, :])
        phi_r_f = np.real(ftf_r[:, fl])
        adj.write_to_files(
            filenames,
            [phi_r_oa, phi_r_f, dfdz, zm_r, np.real(s[0, :]), dfda, a_n, Ir[0, :], Ir[:, fl], np.real(psi)],
        )

    change = np.max(np.abs(adam_step))
    i += 1
    mu = adj.calc_mu(prpt, ptsq)
    if i > stop and mu < 0.5:
        break_flag = True
        break

fr = adj.forward_propagation(k, phi_I, T_R2K, T_K2R, zF, zm_r, k0)
np.savetxt(path + "/data.csv", data, delimiter=",")
final_time = time.time()

if i < steps and not break_flag:
    output = "Algorithm converged in %d steps \nTotal Time:" % (i) + str(
        timedelta(seconds=final_time - start_time)
    )
else:
    output = "Algorithm failed to converge in %d steps \nTotal Time:" % (i) + str(
        timedelta(seconds=final_time - start_time)
    )

print(output)
rr, rt, tt = np.genfromtxt(path + "/data.csv", delimiter=",", unpack=True)
figom = adj.FOM(rr, rt, tt)
mu_val = adj.calc_mu(rt, tt)
plot_fom_mu(path, figom, mu_val)
plot_percent_change_mu(path, mu_val)

with open(path + "/time.txt", "a") as file:
    file.write("\n" + output)

shutil.move(sys.argv[1], path)
print("Finished running main function")
