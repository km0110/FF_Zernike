import sys
import os
import math
import numpy as np
import adjoint_intensity as adj
import time
import shutil
from datetime import timedelta
import sys as _sys, os as _os
_sys.path.append(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "common"))
from plot import plot_fom_mu, plot_percent_change_mu


if len(sys.argv) != 2:
    print("Usage: python fom.py <path to parameter file>")
    sys.exit(1)

# Create the directory for the run
path = sys.argv[1]
params = adj.initialize_variables_from_file(sys.argv[1])
dirname = (path.split("/")[-1])[:-4]
path = path.split("/")[0] + "/runs/int_para2para_" + dirname
if os.path.exists(path):
    shutil.rmtree(path)
os.makedirs(path)  # Calculate all initial parameters from the input file:
subdir = ["I_OA", "I_f", "g", "mirror", "psi_S"]
for s in subdir:
    os.makedirs(path + "/" + s)

with open(path + "/time.txt", "w") as file:
    file.write(str(sys.argv[0]))
################################################
# Lens parameters
f_targ = params["f_targ"]  # focal length of lens
ell = 0  # order of Bessel J function
f_init = params["f_init"]  # initial mirror focal length
conic = params["conic"]

# Beam Parameters
wl = params["wl"]  # wavelength
k0 = 2 * np.pi / wl  # k=2*pi/wavelength
fn = params["f_num"]  # f#
w0 = f_init / (2 * fn)  # initial spot size
rmax = params["r_maxf"] * w0  # the max value of r
nr = params["nr"]  # number of points in r
zR = np.pi * w0**2 / wl  # Initial rayleigh length
wR = 2 / np.pi * fn * wl  # Diffraction limited spot size

# Distance parameters
dz = params["dz"]
nzf = params["nzf"]
zi = f_targ - dz
zf = f_targ + dz
zF = np.linspace(zi, zf, nzf)
fl = len(zF) // 2  # focal length index

# Guess parameters
OoM = math.floor(math.log(w0**2, 10))  # order of magnitude of w0^2
w_guess = 10**OoM  # guess for the spot size
steps = params["steps"]  # max number of steps algorithm is allowed
lr = params["lr"]  # learning rate
tol = params["tol"]  # tolerance
delta = params["delta"]  # weighting function in r
# n = params["n"]
skip = params["skip"]
################################################
print(w0)
# Hankel
r, k, T_R2K, T_K2R = adj.DHTsetup(nr, rmax, ell)
prop_phase = np.exp(-1j * k**2 / (2 * k0) * zF[:, np.newaxis])
adj_phase = prop_phase.T.copy()

# Initial propagation
phi_I = np.exp(-((r / w0) ** 2))
zm_t = adj.create_starting_optic(r, 2 * f_targ)  # target mirror
ftf_t = adj.forward_propagation(
    k, phi_I, T_R2K, T_K2R, zF, zm_t
)  # Target profile defined using target

It = adj.calc_intensity(ftf_t)
maxIt = np.max(It)

It = It / maxIt

# a_n = create_initial_coeff(rmax, 2 * f_init, n)
# print(a_n)
# rho = r / rmax
# Z_fit = calc_Z(rho, n)
# Z_plus = calc_Z_plus(Z_fit)

zm_r = adj.create_starting_optic(r, 2 * f_init)  # initial mirror

data_names = [
    "gradient.csv",
    "mirror.csv",
    "source.csv",
    "int_oa.csv",
    "int_f.csv",
    "psi.csv",
]
filenames = [path + "/" + s for s in data_names]

# Running the main algorithm loop
change = 1000
i = 0
start_time = time.time()
data = []
wt = adj.weight(r, wR, zF, f_targ - dz, f_targ + dz, delta)
while change > tol and i < steps:
    if i % 1 == 0:
        print("MAIN STEP %d \t Grad:%.3e" % (i, change))
    # Forward propagation
    ftf_r = adj.forward_propagation(k, phi_I, T_R2K, T_K2R, zF, zm_r, phase=prop_phase)

    # Calculating Integrals for Source
    Ir = adj.calc_intensity(ftf_r)
    Ir = Ir / maxIt
    Irsq, IrIt, Itsq = adj.int_pairs(r, zF, wt, Ir, It)
    data.append([Irsq, IrIt, Itsq])

    # Calculating S and psi
    s = adj.S(wt, ftf_r, Ir, It, IrIt, Itsq)
    s_hat = T_R2K @ s
    psi_hat = adj.calc_psi_hat(k, k0, zF, s_hat, phase=adj_phase)
    psi = T_K2R @ psi_hat
    # Calculating the gradient and updating the mirror
    dfdz = np.real(adj.gradient(k0, phi_I, psi, zm_r))
    # dfda = dfdz @ Z_fitsure
    # dfda = Z_plus @ dfdz
    # Z_plus = calc_Z_plus(Z_fit)
    # a_n = step(dfda, lr, a_n)
    zm_r = adj.step(dfdz, lr, zm_r)

    # Writing to files
    if i % skip == 0:
        adj.write_to_files(
            filenames,
            [
                dfdz,
                zm_r,
                np.real(s[0, :]),
                Ir[0, :],
                Ir[:, fl],
                np.real(psi),
            ],
        )

    # Updating the change to compare against tolerance
    change = np.max(np.abs(dfdz * lr))
    i += 1
fr = adj.forward_propagation(k, phi_I, T_R2K, T_K2R, zF, zm_r)
np.savetxt(path + "/data.csv", data, delimiter=",")
final_time = time.time()
if i < steps:
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

# write to output file how long it took to run
with open(path + "/time.txt", "a") as file:
    file.write("\n" + output)


# move the parameter file to the run directory
shutil.move(sys.argv[1], path)
print("Finished running main function")
