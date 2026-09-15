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
import cma


def create_axiparabola(r, R, L, f0, taylor=False):
    if taylor:
        sag = r**2 / (4 * f0) * (1 - L / (2 * f0) * (r / R) ** 2)
    else:
        sag = R**2 / (4 * L) * np.log(1 + L / f0 * (r / R) ** 2)
    return sag


def create_flat(r, z, fl, sigma, order):
    R = np.exp(-(((r**2) / (2 * sigma**2)) ** order))[:, np.newaxis]
    Z = np.exp(-((((z - fl - sigma) ** 2) / (sigma**2)) ** order))[np.newaxis, :]
    return R * Z


if len(sys.argv) != 2:
    print("Usage: python main_int_axi2flat_cma.py <path to parameter file>")
    sys.exit(1)

path = sys.argv[1]
params = adj.initialize_variables_from_file(sys.argv[1])
dirname = (path.split("/")[-1])[:-4]
path = path.split("/")[0] + "/runs/int_axi2flat_cma_" + dirname
if os.path.exists(path):
    shutil.rmtree(path)
os.makedirs(path)
subdir = ["I_OA", "I_f", "mirror"]
for s in subdir:
    os.makedirs(path + "/" + s)

with open(path + "/time.txt", "w") as file:
    file.write(str(sys.argv[0]))

################################################
f_targ = params["f_targ"]
ell = 0
f_init = params["f_init"]
conic  = params["conic"]

wl  = params["wl"]
k0  = 2 * np.pi / wl
fn  = params["f_num"]
w0  = f_init / (2 * fn)
rmax = params["r_maxf"] * w0
nr  = params["nr"]
wR  = 2 / np.pi * fn * wl
L   = params["L"]

dz  = params["dz"]
nzf = params["nzf"]
zF  = np.linspace(f_targ - dz, f_targ + L + dz, nzf)
fl  = len(zF) // 2

j   = params["j"]
delta = params["delta"]
sg = params.get("sg", 0)
order = params.get("order", 2)

# CMA-ES parameters
generations = params["steps"]            # number of ask/tell generations (NOT directly comparable to Adam's "steps" -- each generation evaluates a whole population, see popsize below)
sigma0      = params.get("sigma0", 0.3)  # initial step size, in units of each coefficient's own scale (see CMA_stds below)
b           = params.get("b", params.get("lam", 1.0))  # weight on the |1-mu| term in the combined cost (see evaluate() below)
################################################
P = params.get("P", 20)
print(w0)

r, k, T_R2K, T_K2R = adj.DHTsetup(nr, rmax, ell)
prop_phase = np.exp(-1j * k**2 / (2 * k0) * zF[:, np.newaxis])

phi_I = np.exp(-((r / w0) ** 2) ** P)
It = create_flat(r, zF, f_targ, L / 2, 20)

# axi2flat's initial optic is a real axiparabola (not a closed-form 2-mode
# guess like int_para2axi's create_initial_coeff), and the target is an
# abstract flat-top intensity rather than a propagated field -- so maxIt is
# the best-fit scale between the axiparabola's own on-axis intensity and the
# flat target, not a simple peak normalization.
zm_axi  = create_axiparabola(r, w0, L, f_targ)
ftf_axi = adj.forward_propagation(k, phi_I, T_R2K, T_K2R, zF, zm_axi, k0)
It_axi  = adj.calc_intensity(ftf_axi)
maxIt   = np.sum(It_axi[0, :] * It[0, :]) / np.sum(It[0, :] ** 2)

rho    = r / rmax
Z_fit  = adj.calc_Z(rho, j)
Z_plus = adj.calc_Z_plus(Z_fit)

# Initial guess: the Zernike fit of the axiparabola mirror itself, since
# axi2flat starts from a real axiparabola and reshapes it (int_para2axi
# instead approximates a target mirror it doesn't yet have).
a_n = Z_plus @ zm_axi

data_names = ["mirror.csv", "zernike_grad.csv", "zernike_coeff.csv", "int_oa.csv", "int_f.csv"]
filenames = [path + "/" + s for s in data_names]

wt = adj.weight(r, wR, zF, f_targ - dz, f_targ + L + dz, delta, sg, order)


def evaluate(zm_r):
    ftf = adj.forward_propagation(k, phi_I, T_R2K, T_K2R, zF, zm_r, k0, phase=prop_phase)
    Ir  = adj.calc_intensity(ftf) / maxIt
    Irsq, IrIt, Itsq = adj.int_pairs(r, zF, wt, Ir, It)
    fom = adj.FOM(Irsq, IrIt, Itsq)
    mu  = adj.calc_mu(IrIt, Itsq)
    # FOM/Itsq drives shape correlation toward 1 (and keeps the objective in
    # the O(1) range cma's default stopping tolerances assume, rather than
    # FOM's raw O(1e-12)-O(1e-20) scale). On its own, though, FOM/Itsq has a
    # second, degenerate global minimum at Irsq->0 (mirror reflects ~nothing
    # into the weighted region) that scores just as well as a real shape
    # match -- that's the failure mode where mu drifted to 0 instead of 1.
    # The |1-mu| term penalizes that escape route directly.
    cost = fom / Itsq + b * abs(1 - mu)
    return float(cost), Ir, Irsq, IrIt, Itsq


start_time = time.time()
data = []

# CMA-ES adapts a full covariance matrix over the run, but it still needs a
# sane starting scale per coefficient. a_n here is the exact Zernike fit of a
# real axiparabola (not int_para2axi's synthetic 2-mode guess with the rest
# at 0), so every mode already has a genuine nonzero magnitude -- often
# spanning 10+ orders of magnitude from mode 0 to the highest mode. Using
# each coefficient's own magnitude directly (no artificial floor) lets
# CMA-ES explore each mode proportional to its own true scale, rather than
# a fixed floor drowning out how small the high-order modes really need to be.
init_stds = np.abs(a_n)
es = cma.CMAEvolutionStrategy(
    a_n.tolist(),
    sigma0,
    {"CMA_stds": init_stds.tolist(), "maxiter": generations, "verbose": -9},
)

best_a_n  = np.copy(a_n)
best_zm_r = adj.generate_optic(best_a_n, Z_fit)
best_cost, best_Ir, best_Irsq, best_IrIt, best_Itsq = evaluate(best_zm_r)

# Log the unperturbed starting point as generation 0, before es.ask() draws
# any random candidates around it -- otherwise data.csv/mirror.csv's first
# row is already a mutated generation-0 sample, not the true shared starting
# point.
data.append([best_Irsq, best_IrIt, best_Itsq])
adj.write_to_files(
    filenames,
    [best_zm_r, np.zeros_like(best_a_n), best_a_n, best_Ir[0, :], best_Ir[:, fl]],
)

i = 0
while not es.stop():
    # es.mean is CMA-ES's current search mean in Zernike-coefficient space;
    # its shift over this generation (after es.tell() below) is the CMA-ES
    # analog of adam's dfda -- the effective step taken, not just where it
    # ended up (which is what zernike_coeff.csv already records).
    mean_before = np.copy(es.mean)
    candidates = es.ask()

    costs = []
    gen_best_cost = np.inf
    gen_best_Ir = gen_best_Irsq = gen_best_IrIt = gen_best_Itsq = None
    gen_best_zm_r = gen_best_a_n = None

    for a in candidates:
        a = np.asarray(a)
        zm_r = adj.generate_optic(a, Z_fit)
        cost, Ir, Irsq, IrIt, Itsq = evaluate(zm_r)
        costs.append(cost)

        if cost < best_cost:
            best_a_n, best_cost, best_zm_r, best_Ir = np.copy(a), cost, zm_r, Ir

        if cost < gen_best_cost:
            gen_best_cost = cost
            gen_best_Ir, gen_best_Irsq, gen_best_IrIt, gen_best_Itsq = Ir, Irsq, IrIt, Itsq
            gen_best_zm_r, gen_best_a_n = zm_r, a

    es.tell(candidates, costs)
    mean_shift = es.mean - mean_before

    data.append([gen_best_Irsq, gen_best_IrIt, gen_best_Itsq])
    print("GEN %d \t best-of-gen cost=FOM/Itsq+b*|1-mu|:%.3e  overall best:%.3e  sigma=%.2e" %
          (i, gen_best_cost, best_cost, es.sigma))

    adj.write_to_files(
        filenames,
        [gen_best_zm_r, mean_shift, gen_best_a_n, gen_best_Ir[0, :], gen_best_Ir[:, fl]],
    )

    i += 1

np.savetxt(path + "/data.csv", data, delimiter=",")
np.savetxt(path + "/best_zernike_coeff.csv", best_a_n, delimiter=",")
np.savetxt(path + "/best_mirror.csv", best_zm_r, delimiter=",")
np.savetxt(path + "/best_int_oa.csv", best_Ir[0, :], delimiter=",")
np.savetxt(path + "/best_int_f.csv", best_Ir[:, fl], delimiter=",")
final_time = time.time()

output = "CMA-ES finished after %d generations (stop reason: %s), best cost=FOM/Itsq+b*|1-mu|=%.3e \nTotal Time:" % (
    i, list(es.stop().keys()), best_cost) + str(timedelta(seconds=final_time - start_time))

print(output)
rr, rt, tt = np.genfromtxt(path + "/data.csv", delimiter=",", unpack=True)
plot_fom_mu(path, adj.FOM(rr, rt, tt), adj.calc_mu(rt, tt))
plot_percent_change_mu(path, adj.calc_mu(rt, tt))

with open(path + "/time.txt", "a") as file:
    file.write("\n" + output)

shutil.move(sys.argv[1], path)
print("Finished running main function")
