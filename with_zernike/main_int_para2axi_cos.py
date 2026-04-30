import sys
import os
import math
import numpy as np
import adjoint_intensity as adj
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

# Create the directory for the run
path = sys.argv[1]
params = adj.initialize_variables_from_file(sys.argv[1])
dirname = (path.split("/")[-1])[:-4]
path = path.split("/")[0] + "/runs/int_para2axi_cos_" + dirname
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
# w0 = f_targ / (2 * fn)  # initial spot size
w0 = f_init / (2 * fn)  # initial spot size
rmax = params["r_maxf"] * w0  # the max value of r
nr = params["nr"]  # number of points in r
zR = np.pi * w0**2 / wl  # Initial rayleigh length
wR = 2 / np.pi * fn * wl  # Diffraction limited spot size
L = params["L"]

# Distance parameters
dz = params["dz"]
nzf = params["nzf"]
zi = f_targ - dz
zf = f_targ + L + dz
zF = np.linspace(zi, zf, nzf)
fl = len(zF) // 2  # focal length index

# Guess parameters
OoM = math.floor(math.log(w0**2, 10))  # order of magnitude of w0^2
w_guess = 10**OoM  # guess for the spot size
steps = params["steps"]  # max number of steps algorithm is allowed
eta_max = params["lr"]  # learning rate
tol = params["tol"]  # tolerance
delta = params["delta"]  # weighting function in r
n = params["n"]
skip = params["skip"]
stop = params["stop"]
################################################
print(w0)
# Hankel
[r, k] = adj.DHTarrays(nr, rmax, ell)
[T_R2K, T_K2R] = adj.DHTtransformMatrix(nr, rmax, ell)

# Initial propagation
phi_I = np.exp(-((r / w0) ** 40))
zm_t = create_axiparabola(r, w0, L, f_targ)  # target mirror
ftf_t = adj.forward_propagation(
    k, phi_I, T_R2K, T_K2R, zF, zm_t, k0
)  # Target profile defined using target

It = adj.calc_intensity(ftf_t)
maxIt = np.max(It)

It = It / maxIt

# save plot of ftf_t on axis
# print("Plotting ftf_t")
# plt.figure(figsize=(11.52, 6.56), dpi=100)
# plt.plot((zF - f_targ) * 100, np.real(ftf_t[0, :]))
# plt.savefig("check")
# plt.close()
# print("Done")
a_n = adj.create_initial_coeff(rmax, 2 * f_init, n)
print(a_n)
rho = r / rmax
Z_fit = adj.calc_Z(rho, n)
Z_plus = adj.calc_Z_plus(Z_fit)

zm_r = adj.generate_optic(a_n, Z_fit)  # initial mirror

data_names = [
    "gradient.csv",
    "mirror.csv",
    "source.csv",
    "zernike.csv",
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
break_flag = False

# Cosine Annealing Parameters
t_cur = 0
T_0 = 200  # initial cycle length
cycle_mult = 2  # cycle length multiplier
decay_factor = params["decay"]  # decay eta_max after each restart
T_cur = T_0
eta = []
# Adjoint Optimization loop
while change > tol and i < steps:
    if i % 1 == 0:
        print("MAIN STEP %d \t Grad:%.3e" % (i, change))
    # Forward propagation
    ftf_r = adj.forward_propagation(k, phi_I, T_R2K, T_K2R, zF, zm_r, k0)

    # Calculating Integrals for Source
    wt = adj.weight(r, wR, zF, f_targ - dz, f_targ + L + dz, delta)
    Ir = adj.calc_intensity(ftf_r)
    Ir = Ir / maxIt
    Irsq, IrIt, Itsq = adj.int_pairs(r, zF, wt, Ir, It)
    data.append([Irsq, IrIt, Itsq])

    # Calculating S and psi
    s = adj.S(wt, ftf_r, Ir, It, IrIt, Itsq)
    s_hat = T_R2K @ s
    psi_hat = adj.calc_psi_hat(k, k0, zF, s_hat)
    psi = T_K2R @ psi_hat
    # Calculating the gradient and updating the mirror
    dfdz = np.real(adj.gradient(k0, phi_I, psi, a_n, Z_fit))
    # dfda = dfdz @ Z_fit
    dfda = Z_plus @ dfdz
    Z_plus = adj.calc_Z_plus(Z_fit)
    eta_min = eta_max * 1e-2
    eta_t = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * t_cur / T_cur))
    eta.append(eta_t)
    t_cur += 1
    if t_cur >= T_cur:
        t_cur = 0
        T_cur *= cycle_mult  # double the cycle length
        eta_max *= decay_factor  # reduce peak eta_t

    a_n = adj.step(dfda, eta_t, a_n)
    zm_r = adj.generate_optic(a_n, Z_fit)
    # zm_r_unshifted = generate_optic(a_n, Z_fit)
    # zm_r = zm_r_unshifted - zm_r_unshifted[0]

    # Writing to files
    adj.write_to_files(
        filenames,
        [
            dfdz,
            zm_r,
            np.real(s[0, :]),
            dfda,
            Ir[0, :],
            Ir[:, fl],
            np.real(psi),
        ],
    )

    # Updating the change to compare against tolerance
    change = np.sum(np.abs(dfda * eta_t))
    i += 1
    mu = adj.calc_mu(IrIt, Itsq)
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

# write to output file how long it took to run
with open(path + "/time.txt", "a") as file:
    file.write("\n" + output)


# move the parameter file to the run directory
shutil.move(sys.argv[1], path)
print("Finished running main function")
