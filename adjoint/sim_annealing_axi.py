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


if len(sys.argv) != 2:
    print("Usage: python fom.py <path to parameter file>")
    sys.exit(1)

# Create the directory for the run
path = sys.argv[1]
params = initialize_variables_from_file(sys.argv[1])
dirname = (path.split("/")[-1])[:-4]
path = path.split("/")[0] + "/runs/real_" + dirname
if os.path.exists(path):
    shutil.rmtree(path)
os.makedirs(path)  # Calculate all initial parameters from the input file:
subdir = ["phr_f", "pr_oa", "pr_oa_I", "g", "psi", "mirror"]
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
lr = params["lr"]  # learning rate
tol = params["tol"]  # tolerance
delta = params["delta"]  # weighting function in r
n = params["n"]
skip = params["skip"]
stop = params["stop"]
temp = params["temp"]
c = params["c"]
floor = params["floor"]

################################################
print(w0)
# Hankel
[r, k] = DHTarrays(nr, rmax, ell)
[T_R2K, T_K2R] = DHTtransformMatrix(nr, rmax, ell)


data_names = [
    "phi_r_OA.csv",
    "phi_r_OA_abs.csv",
    "phi_r_focus.csv",
    "gradient.csv",
    "mirror.csv",
    "source.csv",
    "zernike.csv",
]
filenames = [path + "/" + s for s in data_names]

# Running the main algorithm loop
change = 1000
i = 1
start_time = time.time()
data = []

# Initial propagation
phi_I = 1j * np.exp(-((r / w0) ** 40))
zm_t = create_axiparabola(r, w0, L, f_targ)  # target mirror
ftf_t = forward_propagation(
    k, phi_I, T_R2K, T_K2R, zF, zm_t
)  # Target profile defined using target mirror

best_a_n = create_initial_coeff(rmax, 2 * f_init, n)
rho = r / rmax
Z_fit = calc_Z(rho, n)
Z_plus = calc_Z_plus(Z_fit)

curr_zm_r = generate_optic(best_a_n, Z_fit)  # initial mirror
curr_ftf_r = forward_propagation(k, phi_I, T_R2K, T_K2R, zF, curr_zm_r)

# Calculating Integrals for Source
curr_pR, curr_pT = weighted_phi(
    r, wR, zF, f_targ - dz, f_targ + dz, curr_ftf_r, ftf_t, delta
)
curr_prsq, curr_prpt, curr_ptsq = int_pairs(
    r, zF, curr_pR, curr_pT, curr_ftf_r, ftf_t
)  # phi_r_sq, phi_r_phi_t, phi_t_sq
best_fom = FOM(curr_prsq, curr_prpt, curr_ptsq)
curr_a_n, curr_fom = best_a_n, best_fom
new_count = 0
stagnation_counter = 0  # Track number of iterations with no significant improvement
stagnation_limit = 10  # Maximum stagnation iterations before adjustment
data.append(np.real([curr_prsq, curr_prpt, curr_ptsq]))
del_improv = lr / 10
del_stag = lr / 100
break_flag = False
lrtxt = []
while change > tol and i <= steps:
    print("MAIN STEP %d \t Grad:%.3e" % (i, change))
    # Calculating S and psi
    s = S(curr_pR, curr_pT, curr_prpt, curr_ptsq)
    s_hat = T_R2K @ s
    psi_hat = calc_psi_hat(k, k0, zF, s_hat)
    psi = T_K2R @ psi_hat

    # Calculating the gradient and updating the mirror
    dfdz = np.real(gradient(k0, phi_I, psi, curr_a_n, Z_fit))
    # dfda = dfdz @ Z_fit
    dfda = Z_plus @ dfdz
    Z_plus = calc_Z_plus(Z_fit)

    # Writing to files
    phi_r_oa = np.real(curr_ftf_r[0, :])
    phi_r_oa_abs = np.abs(curr_ftf_r[0, :])
    phi_r_f = np.real(curr_ftf_r[:, fl])

    # calculate the candidate mirror and the FOM score for it
    cand_a_n = step(dfda, np.random.random() * lr, curr_a_n)
    cand_zm_r = generate_optic(cand_a_n, Z_fit)

    cand_ftf_r = forward_propagation(k, phi_I, T_R2K, T_K2R, zF, cand_zm_r)

    # Calculating Integrals for Source
    cand_pR, cand_pT = weighted_phi(
        r, wR, zF, f_targ - dz, f_targ + dz, cand_ftf_r, ftf_t, delta
    )
    cand_prsq, cand_prpt, cand_ptsq = int_pairs(
        r, zF, cand_pR, cand_pT, cand_ftf_r, ftf_t
    )  # phi_r_sq, phi_r_phi_t, phi_t_sq
    cand_fom = FOM(cand_prsq, cand_prpt, cand_ptsq)

    if cand_fom < curr_fom:
        best_a_n, best_fom_fom = cand_a_n, cand_fom
        data.append(np.real([cand_prsq, cand_prpt, cand_ptsq]))

    diff = cand_fom - curr_fom
    t = c * temp / np.log(1 + i)
    if t < floor:
        curr_t = floor
        # write to text file current iteration
    else:
        curr_t = t

    write_to_files(
        filenames,
        [phi_r_oa, phi_r_oa_abs, phi_r_f, dfdz, curr_zm_r, np.real(psi), dfda],
    )

    lrtxt.append(curr_t)

    metropolis = np.exp(-diff / curr_t)
    if diff < 0 or np.random.random() < metropolis:
        curr_a_n, curr_fom = cand_a_n, cand_fom
        curr_zm_r = cand_zm_r
        curr_ftf_r = cand_ftf_r
        curr_pR, curr_pT = cand_pR, cand_pT
        curr_prsq, curr_prpt, curr_ptsq = cand_prsq, cand_prpt, cand_ptsq
        # if diff < 0 and np.abs(diff)>del_improv:
        #     c *= 1.5
        # elif diff < del_stag:
        #     stagnation_counter += 1
        #     if stagnation_counter > stagnation_limit:
        #         c /= 2
        #         stagnation_counter = 0
        # else:
        #     stagnation_counter = 0

        new_count += 1
    i += 1
    mu = calc_mu(curr_prpt, curr_ptsq)

    if i > stop and mu < 0.5:
        break_flag = True
        break

    # # Writing to files
    # phi_r_oa = np.real(curr_ftf_r[0, :])
    # phi_r_oa_abs = np.abs(curr_ftf_r[0, :])
    # phi_r_f = np.real(curr_ftf_r[:, fl])
    # write_to_files(
    #     filenames,
    #     [phi_r_oa, phi_r_oa_abs, phi_r_f, dfdz, curr_zm_r, np.real(psi), dfda],
    # )

    # Updating the change to compare against tolerance
    change = np.sum(np.abs(dfda * lr))
np.savetxt(path + "/data.csv", data, delimiter=",")
np.savetxt(path + "/lr.csv", lrtxt, delimiter=",")

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
phr_oa, phr_oa_abs, phr_f, gr, mr, psis, zern = [
    np.genfromtxt(f, delimiter=",") for f in filenames
]
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
plot_phi_t_phi_r_f(path, phr_f, r, w0, np.real(ftf_t), fl, skip)
prft = time.time()
print("PRPT_F Time:" + str(timedelta(seconds=prft - pplots)))

plot_phi_t_phi_r_OA(path, phr_oa, zF, f_targ, np.real(ftf_t), skip)
proat = time.time()
print("PRPT_OA Time:" + str(timedelta(seconds=proat - prft)))

plot_phi_t_phi_r_OA_I(path, phr_oa_abs, zF, f_targ, np.abs(ftf_t), skip)

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

# move the parameter file to the run directory
shutil.move(sys.argv[1], path)
