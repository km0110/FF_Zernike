import sys
import os
from gaussian_beam import *
import importlib
import adjoint_intensity as adj
# print("Using:", adjoint_intensity.__file__)
# importlib.reload(adjoint_intensity)
# from adjoint_intensity import *
from plot import *
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
path = path.split("/")[0] + "/runs/int_axi_" + dirname
if os.path.exists(path):
    shutil.rmtree(path)
os.makedirs(path)  # Calculate all initial parameters from the input file:
subdir = ["phr_f", "pr_oa", "I_OA", "I_f", "g", "psi_S", "mirror"]
for s in subdir:
    os.makedirs(path + "/" + s)

with open(path + "/time.txt", "w") as file:
    file.write(sys.argv[0])

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
fl = len(zF) // 4  # focal length index

# Guess parameters
OoM = math.floor(math.log(w0**2, 10))  # order of magnitude of w0^2
w_guess = 10**OoM  # guess for the spot size
steps = params["steps"]  # max number of steps algorithm is allowed
lr = params["lr"]  # learning rate
tol = params["tol"]  # tolerance
delta = params["delta"]  # weighting function in r
num = params["n"]
skip = params["skip"]
stop = params["stop"]
# power = params['power']
################################################
power=40
print(w0)
# Hankel
[r, k] = adj.DHTarrays(nr, rmax, ell)
[T_R2K, T_K2R] = adj.DHTtransformMatrix(nr, rmax, ell)

# Initial propagation
phi_I = 1j * np.exp(-((r / w0) ** power))
zm_t = create_axiparabola(r, w0, L, f_targ)  # target mirror
ftf_t = adj.forward_propagation(
    k, phi_I, T_R2K, T_K2R, zF, zm_t
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
a_n = adj.create_initial_coeff(rmax, 2 * f_init, num)
rho = r / rmax
Z_fit = adj.calc_Z(rho, num)
Z_plus = adj.calc_Z_plus(Z_fit)

zm_r = adj.generate_optic(a_n, Z_fit)  # initial mirror

data_names = [
    "phi_r_OA.csv",
    "phi_r_focus.csv",
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
while change > tol and i < steps+1:
    if i % 1 == 0:
        print("MAIN STEP %d \t Grad:%.3e" % (i, change))
    # Forward propagation
    ftf_r = adj.forward_propagation(k, phi_I, T_R2K, T_K2R, zF, zm_r)

    # Calculating Integrals for Source
    wt = adj.weight(r, wR, zF, f_targ - dz, f_targ + L + dz, delta)
    Ir = adj.calc_intensity(ftf_r)
    Ir = Ir / maxIt
    Irsq, IrIt, Itsq = adj.int_pairs(r, zF, wt, Ir, It)
    data.append(np.real([Irsq, IrIt, Itsq]))

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
    a_n = adj.step(dfda, lr, a_n)
    zm_r = adj.generate_optic(a_n, Z_fit)
    # zm_r_unshifted = generate_optic(a_n, Z_fit)
    # zm_r = zm_r_unshifted - zm_r_unshifted[0]

    # Writing to files
    phi_r_oa = np.real(ftf_r[0, :])
    phi_r_f = np.real(ftf_r[:, fl])
    adj.write_to_files(
        filenames,
        [
            phi_r_oa,
            phi_r_f,
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
    change = np.sum(np.abs(dfda * lr))
    i += 1
    mu = adj.calc_mu(IrIt, Itsq)
    if i > stop and mu < 0.5:
        break_flag = True
        break

fr = adj.forward_propagation(k, phi_I, T_R2K, T_K2R, zF, zm_r)
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

# write to output file how long it took to run
with open(path + "/time.txt", "w") as file:
    file.write(output)

################################################
# create plots
phr_oa, phr_f, gr, mr, src, zern, ir_oa, ir_f, psis = [
    np.genfromtxt(f, delimiter=",") for f in filenames
]
pr2, rt, pt2 = np.genfromtxt(path + "/data.csv", delimiter=",", unpack=True)
figom = adj.FOM(pr2, rt, pt2)
mu_val = adj.calc_mu(rt, pt2)
print("START PLOTTING")
pstart = time.time()

plot_total_change(path, r, w0, zern, lr, wl, i, num)
print("1")
plot_max_grad(path, gr)
print("2")
plot_fom_mu(path, figom, mu_val)
print("3")
plot_integrals(path, pr2, rt, pt2)
plot_mirror_comp(path, r, w0, zm_t, mr)
plot_zernikes(path, zern, rmax, f_targ, num)

pplots = time.time()
# print("Plotting Time:" + str(timedelta(seconds=pplots - pstart)))
plot_mirror_comp_each(path, r, w0, zm_t, mr, skip)

# Plotting each iteration phi_r and phi_t, the gradient and the source
# plot_phi_t_phi_r_f(path, phr_f, r, w0, ftf_t, fl, skip)
# prft = time.time()
# print("PRPT_F Time:" + str(timedelta(seconds=prft - pplots)))

# plot_phi_t_phi_r_OA(path, phr_oa, zF, f_targ, ftf_t, skip)
# proat = time.time()
# print("PRPT_OA Time:" + str(timedelta(seconds=proat - prft)))

plot_I_OA(path, ir_oa, zF, f_targ, It[0, :], skip)

plot_I_f(path, ir_f, r, w0, It[:, fl], skip)

# plot_da_vs_grad(path, gr, r, w0, zern, n, skip)
# pdzt = time.time()
# print("G Time:" + str(timedelta(seconds=pdzt - proat)))


# plot_psi(path, psis, r, w0, skip)
# ps = time.time()
# print("Psi Time:" + str(timedelta(seconds=ps - pdzt)))

# plot_source(path, src, zF, f_targ, skip)
plot_psi_and_source(path, psis, r, w0, src, zF, f_targ, skip)

pend = time.time()

print("Total Time:" + str(timedelta(seconds=pend - pstart)))

##add to time.txt the time it took to plot
with open(path + "/time.txt", "a") as file:
    file.write("\nPlotting Time:" + str(timedelta(seconds=pend - pstart)))

# move the parameter file to the run directory
shutil.move(sys.argv[1], path)
