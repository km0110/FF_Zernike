import sys
import os
from gaussian_beam import *
from adjoint import *
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


if len(sys.argv) != 4:
    print(
        "Usage: python cont_main_axi.py <path to run folder> <number of additional steps> <learning rate>"
    )
    sys.exit(1)

# Create the directory for the run
original_path = sys.argv[1]
param_file = (original_path.split("/")[-2])[5:] + ".txt"
print(param_file)
params = initialize_variables_from_file(original_path + param_file)
dirname = original_path.split("/")[-2]
path = original_path.split("/")[0] + "/runs/" + dirname + "_cont_1"
if os.path.exists(path):
    shutil.rmtree(path)
os.makedirs(path)  # Calculate all initial parameters from the input file:
subdir = ["phr_f", "pr_oa", "pr_oa_I", "g", "psi", "mirror"]
for s in subdir:
    os.makedirs(path + "/" + s)

cont = int(sys.argv[2])
###########################################  #####
# Lens parameters
f_targ = params["f_targ"]  # focal length of lens
ell = 0  # order of Bessel J function
f_init = params["f_init"]  # initial mirror focal length
conic = params["conic"]

# Beam Parameters
wl = params["wl"]  # wavelength
k0 = 2 * np.pi / wl  # k=2*pi/wavelength
fn = params["f_num"]  # f#
w0 = f_targ / (2 * fn)  # initial spot size
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
lr = float(sys.argv[3])  # learning rate
tol = params["tol"]  # tolerance
delta = params["delta"]  # weighting function in r
n = params["n"]
skip = params["skip"]
stop = params["stop"]
################################################
print(w0)
# Hankel
[r, k] = DHTarrays(nr, rmax, ell)
[T_R2K, T_K2R] = DHTtransformMatrix(nr, rmax, ell)

# Initial propagation
phi_I = 1j * np.exp(-((r / w0) ** 40))
zm_t = create_axiparabola(r, w0, L, f_targ)  # target mirror
ftf_t = forward_propagation(
    k, phi_I, T_R2K, T_K2R, zF, zm_t
)  # Target profile defined using target

data_names = [
    "phi_r_OA.csv",
    "phi_r_focus.csv",
    "gradient.csv",
    "mirror.csv",
    "source.csv",
    "zernike.csv",
]
filenames = [path + "/" + s for s in data_names]

rho = r / rmax
Z_fit = calc_Z(rho, n)
Z_plus = calc_Z_plus(Z_fit)

# take the last mirror profile as the initial guess

zm_r = np.genfromtxt(original_path + "mirror.csv", delimiter=",")[-1]
a_n = np.genfromtxt(original_path + "zernike.csv", delimiter=",")[-1]
print(len(zm_r))
# Running the main algorithm loop
change = 1000
i = steps
start_time = time.time()
data = []
break_flag = False
while change > tol and i < steps + cont:
    if i % 1 == 0:
        print("MAIN STEP %d \t Grad:%.3e" % (i, change))
    # Forward propagation
    ftf_r = forward_propagation(k, phi_I, T_R2K, T_K2R, zF, zm_r)

    # Calculating Integrals for Source
    pR, pT = weighted_phi(r, wR, zF, f_targ - dz, f_targ + L + dz, ftf_r, ftf_t, delta)
    prsq, prpt, ptsq = int_pairs(
        r, zF, pR, pT, ftf_r, ftf_t
    )  # phi_r_sq, phi_r_phi_t, phi_t_sq
    data.append(np.real([prsq, prpt, ptsq]))

    # Calculating S and psi
    s = S(pR, pT, prpt, ptsq)
    s_hat = T_R2K @ s
    psi_hat = calc_psi_hat(k, k0, zF, s_hat)
    psi = T_K2R @ psi_hat
    # Calculating the gradient and updating the mirror
    dfdz = np.real(gradient(k0, phi_I, psi, a_n, Z_fit))
    # dfda = dfdz @ Z_fit
    dfda = Z_plus @ dfdz
    Z_plus = calc_Z_plus(Z_fit)
    a_n = step(dfda, lr, a_n)
    zm_r = generate_optic(a_n, Z_fit)

    # Writing to files
    phi_r_oa = np.real(ftf_r[0, :])
    phi_r_f = np.real(ftf_r[:, fl])
    write_to_files(filenames, [phi_r_oa, phi_r_f, dfdz, zm_r, np.real(psi), dfda])

    # Updating the change to compare against tolerance
    change = np.sum(np.abs(dfda * lr))
    i += 1

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
phr_oa, phr_f, gr, mr, psis, zern = [np.genfromtxt(f, delimiter=",") for f in filenames]
pr2, rt, pt2 = np.genfromtxt(path + "/data.csv", delimiter=",", unpack=True)
figom = FOM(pr2, rt, pt2)
mu_val = calc_mu(rt, pt2)
print("START PLOTTING")
pstart = time.time()

print("psi:", psis.shape)
plot_total_change(path, r, w0, zern, lr, wl, i, n)
print("1")
plot_max_grad(path, gr)
print("2")
plot_fom_mu(path, figom, mu_val)
print("3")
plot_integrals(path, pr2, rt, pt2)
plot_mirror_comp(path, r, w0, zm_t, mr, i)
plot_zernikes(path, zern, rmax, f_targ, n)

pplots = time.time()
# print("Plotting Time:" + str(timedelta(seconds=pplots - pstart)))
plot_mirror_comp_each(path, r, w0, zm_t, mr, skip)

# Plotting each iteration phi_r and phi_t, the gradient and the source
plot_phi_t_phi_r_f(path, phr_f, r, w0, ftf_t, fl, skip)
prft = time.time()
print("PRPT_F Time:" + str(timedelta(seconds=prft - pplots)))

plot_phi_t_phi_r_OA(path, phr_oa, zF, f_targ, ftf_t, skip)
proat = time.time()
print("PRPT_OA Time:" + str(timedelta(seconds=proat - prft)))

plot_phi_t_phi_r_OA_I(path, phr_oa, zF, f_targ, ftf_t, skip)

plot_da_vs_grad(path, gr, r, w0, zern, n, skip)
pdzt = time.time()
print("G Time:" + str(timedelta(seconds=pdzt - proat)))


plot_psi(path, psis, r, w0, skip)
ps = time.time()
print("Psi Time:" + str(timedelta(seconds=ps - pdzt)))

pend = time.time()

print("Total Time:" + str(timedelta(seconds=pend - pstart)))

##add to time.txt the time it took to plot
with open(path + "/time.txt", "a") as file:
    file.write("\nPlotting Time:" + str(timedelta(seconds=pend - pstart)))
