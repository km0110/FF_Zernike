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
import multiprocessing
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
    print("Usage: python main_int_para2flat_cma.py <path to parameter file>")
    sys.exit(1)

path = sys.argv[1]
params = adj.initialize_variables_from_file(sys.argv[1])
dirname = (path.split("/")[-1])[:-4]
path = path.split("/")[0] + "/runs/int_para2flat_cma_" + dirname
if os.path.exists(path):
    shutil.rmtree(path)
os.makedirs(path)

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

# CMA-ES parameters
generations = params["steps"]            # number of ask/tell generations PER RESTART (see restarts below)
sigma0      = params.get("sigma0", 0.3)  # initial step size, in units of each coefficient's own scale (see CMA_stds below)
b           = params.get("b", params.get("lam", 1.0))  # weight on the |1-mu| term in the combined cost (see evaluate() below)
# Same rationale as int_para2axi's restart loop: the FOM/Itsq landscape has
# multiple well-separated local optima in Zernike-coefficient space, so a
# single CMA-ES run can't know which basin it landed in. Run several
# independent restarts (fresh random seed each time, same full generation
# budget each) and keep whichever one reaches the lowest cost.
restarts = int(params.get("restarts", 5))
# Lets multiple separate invocations of this script (e.g. one per OS process,
# for external parallelism) each claim a disjoint block of seeds instead of
# all restarting from seed=1 -- see restart loop below.
seed_offset = int(params.get("seed_offset", 0))
# Restarts are independent (each seeds its own CMA-ES instance from scratch),
# so they run in a fork-based multiprocessing pool rather than one at a time.
# Fork (not the macOS-default spawn) is used deliberately: it hands each
# worker the already-computed physics setup below (DHT operators, Z_fit,
# etc.) via copy-on-write instead of re-running this whole module per worker.
workers = int(params.get("workers", min(restarts, os.cpu_count() or 1)))
################################################
power = 40
print(w0)

r, k, T_R2K, T_K2R = adj.DHTsetup(nr, rmax, ell)
prop_phase = np.exp(-1j * k**2 / (2 * k0) * zF[:, np.newaxis])

phi_I = np.exp(-((r / w0) ** power))
It = create_flat(r, zF, f_targ, L / 2, 20)

# Same normalization scheme as axi2flat: It is an abstract flat-top shape
# with no physical intensity scale of its own, so maxIt is the best-fit
# scale between a real axiparabola's on-axis intensity and that abstract
# shape. This is a fixed physical reference, deliberately independent of
# a_n below -- tying maxIt to the actual starting mirror instead (as an
# earlier version of this file did) pins mu to ~1 right from generation 0
# regardless of true shape match, since the same optic sets both, and that
# hides any real optimization signal in the mu/FOM diagnostics.
zm_axi  = create_axiparabola(r, w0, L, f_targ)
ftf_axi = adj.forward_propagation(k, phi_I, T_R2K, T_K2R, zF, zm_axi, k0)
It_axi  = adj.calc_intensity(ftf_axi)
maxIt   = np.sum(It_axi[0, :] * It[0, :]) / np.sum(It[0, :] ** 2)

# para2flat's actual starting guess is still the closed-form 2-mode parabola
# from int_para2axi (not the axiparabola above, which is only a normalization
# reference).
a_n    = adj.create_initial_coeff(rmax, 2 * f_init, j)
rho    = r / rmax
Z_fit  = adj.calc_Z(rho, j)

wt = adj.weight(r, wR, zF, f_targ - dz, f_targ + L + dz, delta)


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


def run_one_cma(seed, run_path):
    """Runs one full CMA-ES optimization to completion, writing all its
    per-generation outputs to run_path in the same layout the script used to
    write directly to its top-level output folder. Returns a summary dict
    used to compare this restart against the others.
    """
    print("=== starting restart seed=%d ===" % seed)
    for s in ["I_OA", "I_f", "mirror"]:
        os.makedirs(run_path + "/" + s, exist_ok=True)

    data_names = ["mirror.csv", "zernike_grad.csv", "zernike_coeff.csv", "int_oa.csv", "int_f.csv"]
    filenames = [run_path + "/" + s for s in data_names]

    start_time = time.time()
    data = []

    # CMA-ES adapts a full covariance matrix over the run, but it still needs
    # a sane starting scale per coefficient. The Zernike coefficients here
    # span ~2 orders of magnitude across modes (a_0/a_1 are O(1e-5), higher
    # modes settle ~100x smaller -- see create_initial_coeff, which only
    # seeds modes 0/1 and leaves the rest at 0). CMA_stds gives CMA-ES that
    # per-coordinate scale up front so it isn't exploring every mode with the
    # same absolute step size, which is exactly what broke the fixed-lr SA
    # random walk.
    init_stds = np.maximum(np.abs(a_n), np.abs(a_n[0]) * 1e-2)
    es = cma.CMAEvolutionStrategy(
        a_n.tolist(),
        sigma0,
        {"CMA_stds": init_stds.tolist(), "maxiter": generations, "verbose": -9, "seed": seed},
    )

    best_a_n  = np.copy(a_n)
    best_zm_r = adj.generate_optic(best_a_n, Z_fit)
    best_cost, best_Ir, best_Irsq, best_IrIt, best_Itsq = evaluate(best_zm_r)

    # Log the unperturbed starting point as generation 0, before es.ask()
    # draws any random candidates around it -- otherwise data.csv/mirror.csv's
    # first row is already a mutated generation-0 sample, not the true shared
    # starting point (this is what made mirror_comp.pdf's "Initial Mirror"
    # look wrong).
    data.append([best_Irsq, best_IrIt, best_Itsq])
    adj.write_to_files(
        filenames,
        [best_zm_r, np.zeros_like(best_a_n), best_a_n, best_Ir[0, :], best_Ir[:, fl]],
    )

    i = 0
    while not es.stop():
        # es.mean is CMA-ES's current search mean in Zernike-coefficient
        # space; its shift over this generation (after es.tell() below) is
        # the CMA-ES analog of adam's dfda -- the effective step taken, not
        # just where it ended up (which is what zernike_coeff.csv already
        # records).
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
        print("  [seed %d] GEN %d \t best-of-gen cost=FOM/Itsq+b*|1-mu|:%.3e  overall best:%.3e  sigma=%.2e" %
              (seed, i, gen_best_cost, best_cost, es.sigma))

        adj.write_to_files(
            filenames,
            [gen_best_zm_r, mean_shift, gen_best_a_n, gen_best_Ir[0, :], gen_best_Ir[:, fl]],
        )

        i += 1

    np.savetxt(run_path + "/data.csv", data, delimiter=",")
    np.savetxt(run_path + "/best_zernike_coeff.csv", best_a_n, delimiter=",")
    np.savetxt(run_path + "/best_mirror.csv", best_zm_r, delimiter=",")
    np.savetxt(run_path + "/best_int_oa.csv", best_Ir[0, :], delimiter=",")
    np.savetxt(run_path + "/best_int_f.csv", best_Ir[:, fl], delimiter=",")
    elapsed = time.time() - start_time

    stop_reason = list(es.stop().keys())
    run_output = "CMA-ES finished after %d generations (stop reason: %s), best cost=FOM/Itsq+b*|1-mu|=%.3e \nTotal Time:" % (
        i, stop_reason, best_cost) + str(timedelta(seconds=elapsed))
    with open(run_path + "/time.txt", "w") as file:
        file.write(run_output)

    rr, rt, tt = np.genfromtxt(run_path + "/data.csv", delimiter=",", unpack=True)
    plot_fom_mu(run_path, adj.FOM(rr, rt, tt), adj.calc_mu(rt, tt))
    plot_percent_change_mu(run_path, adj.calc_mu(rt, tt))

    return {
        "seed": seed,
        "best_cost": best_cost,
        "generations": i,
        "stop_reason": stop_reason,
        "elapsed": elapsed,
        "run_path": run_path,
    }


start_time = time.time()
jobs = [(seed_offset + i + 1, path + "/restart_%d" % i) for i in range(restarts)]
print("Running %d restarts across %d worker process(es)..." % (restarts, workers))
with multiprocessing.get_context("fork").Pool(processes=workers) as pool:
    restart_summaries = pool.starmap(run_one_cma, jobs)

best_restart = min(restart_summaries, key=lambda s: s["best_cost"])
final_time = time.time()

# Promote the winning restart's files to the top level, matching the
# original single-run script's output layout exactly, so downstream tools
# (make_plots.py etc.) keep working unchanged.
for item in os.listdir(best_restart["run_path"]):
    src = os.path.join(best_restart["run_path"], item)
    dst = os.path.join(path, item)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)

summary_lines = ["restart\tseed\tgenerations\tstop_reason\tbest_cost"]
for idx, s in enumerate(restart_summaries):
    marker = "\t<== winner" if s is best_restart else ""
    summary_lines.append("%d\t%d\t%d\t%s\t%.6e%s" % (
        idx, s["seed"], s["generations"], s["stop_reason"], s["best_cost"], marker))
with open(path + "/restarts_summary.txt", "w") as f:
    f.write("\n".join(summary_lines))

output = ("%d restarts finished. Winning restart: seed=%d, best cost=FOM/Itsq+b*|1-mu|=%.3e \nTotal Time:" % (
    restarts, best_restart["seed"], best_restart["best_cost"])) + str(timedelta(seconds=final_time - start_time))

print(output)
with open(path + "/time.txt", "a") as file:
    file.write("\n" + output)

shutil.move(sys.argv[1], path)
print("Finished running main function")
