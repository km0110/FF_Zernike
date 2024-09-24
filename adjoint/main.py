import sys
import os
from gaussian_beam import *
from adjoint import *
from plot import *
import time
import shutil
from datetime import timedelta

if len(sys.argv) != 2:
    print("Usage: python fom.py <path to parameter file>")
    sys.exit(1)

# Create the directory for the run
path = sys.argv[1]
params = initialize_variables_from_file(sys.argv[1])
dirname = (path.split("/")[-1])[:-4]
path = "runs/" + dirname
if os.path.exists(path):
    shutil.rmtree(path)
os.makedirs(path)  # Calculate all initial parameters from the input file:
subdir = ["phr_f", "pr_oa", "g", "s", "mirror"]
for s in subdir:
    os.makedirs(path + "/" + s)

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
zF = np.linspace(f_targ - dz, f_targ + dz, nzf)
fl = len(zF) // 2  # focal length index

# Guess parameters
OoM = math.floor(math.log(w0**2, 10))  # order of magnitude of w0^2
w_guess = 10**OoM  # guess for the spot size
steps = params["steps"]  # max number of steps algorithm is allowed
lr = params["lr"]  # learning rate
tol = params["tol"]  # tolerance
delta = params["delta"]  # weighting function in r

################################################
print(w0)
# Hankel
[r, k] = DHTarrays(nr, rmax, ell)
[T_R2K, T_K2R] = DHTtransformMatrix(nr, rmax, ell)

# Initial propagation
phi_I = 1j * np.exp(-((r / w0) ** 2))
zm_t = create_starting_optic(r, 2 * f_targ)  # target mirror
ftf_t = forward_propagation(
    k, phi_I, T_R2K, T_K2R, zF, zm_t
)  # Target profile defined using target mirror

a_n = create_initial_coeff(rmax, 2 * f_init)
print(a_n)
rho = r / rmax
Z_fit = calc_Z(rho)
Z_plus = calc_Z_plus(Z_fit)

zm_r = generate_optic(a_n, Z_fit)  # initial mirror

data_names = [
    "phi_r_OA.csv",
    "phi_r_focus.csv",
    "gradient.csv",
    "mirror.csv",
    "source.csv",
    "zernike.csv",
]
filenames = [path + "/" + s for s in data_names]

# Running the main algorithm loop
change = 1000
i = 0
start_time = time.time()
data = []
while change > tol and i < steps:
    if i % 1 == 0:
        print("MAIN STEP %d \t Grad:%.3e" % (i, change))
    # Forward propagation
    ftf_r = forward_propagation(k, phi_I, T_R2K, T_K2R, zF, zm_r)

    # Calculating Integrals for Source
    pR, pT = weighted_phi(r, wR, zF, f_targ - dz, f_targ + dz, ftf_r, ftf_t, delta)
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
    write_to_files(filenames, [phi_r_oa, phi_r_f, dfdz, zm_r, np.real(s[0, :]), dfda])

    # Updating the change to compare against tolerance
    change = np.sum(np.abs(dfda * lr))
    i += 1
fr = forward_propagation(k, phi_I, T_R2K, T_K2R, zF, zm_r)
np.savetxt(path + "/data.csv", data, delimiter=",")
final_time = time.time()
if i < steps:
    print(
        "Algorithm converged in %d steps \nTotal Time:" % (i)
        + str(timedelta(seconds=final_time - start_time))
    )
else:
    print(
        "Algorithm failed to converge in %d steps \nTotal Time:" % (i)
        + str(timedelta(seconds=final_time - start_time))
    )

################################################
# create plots
phr_oa, phr_f, gr, mr, src, zern = [np.genfromtxt(f, delimiter=",") for f in filenames]
pr2, rt, pt2 = np.genfromtxt(path + "/data.csv", delimiter=",", unpack=True)
figom = FOM(pr2, rt, pt2)
mu_val = calc_mu(rt, pt2)
print("START PLOTTING")
pstart = time.time()

# Plotting overall change
plot_total_change(path, r, w0, zern, lr, wl, i, xlim=(0, 2))
print("1")
plot_max_grad(path, gr)
print("2")
plot_fom_mu(path, figom, mu_val)
print("3")
plot_integrals(path, pr2, rt, pt2)
plot_mirror_comp(path, r, w0, zm_t, mr, i)
plot_mirror_comp_each(path, r, w0, zm_t, mr)
plot_mirror_all(path, r, w0, mr)

plot_zernikes(path, zern, rmax, f_targ)
# plot_mirror_comp(path, r, w0, zm_t, mr, i, conic, do_shift=True)
# plot_mirror_be_af(path, r, w0, mr)
pplots = time.time()
print("Plotting Time:" + str(timedelta(seconds=pplots - pstart)))

# Plotting each iteration phi_r and phi_t, the gradient and the source
plot_phi_t_phi_r_f(path, phr_f, r, w0, ftf_t, fl)
prft = time.time()
print("PRPT_F Time:" + str(timedelta(seconds=prft - pplots)))

plot_phi_t_phi_r_OA(path, phr_oa, zF, f_targ, ftf_t)
proat = time.time()
print("PRPT_OA Time:" + str(timedelta(seconds=proat - prft)))

# plot_da_vs_grad(path, lr, gr, r, w0, wl, zern)
# pdzt = time.time()
# print("G Time:" + str(timedelta(seconds=pdzt - proat)))

# plot_s(path, src, zF, f_targ)
# ps = time.time()
# print("S Time:" + str(timedelta(seconds=ps - pdzt)))

pend = time.time()

print("Total Time:" + str(timedelta(seconds=pend - pstart)))
