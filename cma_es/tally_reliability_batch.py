#!/usr/bin/env python
"""
Tallies a completed reliability-study batch (see generate_reliability_batch.py
/ run_reliability_batch.sh): for every completed run, recomputes its true
cost directly from its saved best_zernike_coeff.csv (not parsed from a log
file), auto-clusters the results by cost (biggest gap in log10-cost splits
"target basin" from "everything else" -- doesn't assume there are exactly 2
basins), and reports the resulting success rate with a Wilson-score 95%
confidence interval.

Usage:
    python tally_reliability_batch.py [batch_dir]

batch_dir defaults to "reliability_study_n100".
"""

import os
import sys
import glob
import numpy as np
import sys as _sys, os as _os
_sys.path.append(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "common"))
import adjoint_intensity_zernike as adj

batch_dir = sys.argv[1] if len(sys.argv) >= 2 else "reliability_study_n100"
runs_dir = os.path.join(batch_dir, "runs")


def create_axiparabola(r, R, L, f0):
    return R**2 / (4 * L) * np.log(1 + L / f0 * (r / R) ** 2)


def generate_optic(a_n, zernpol):
    zm = adj.generate_optic(a_n, zernpol)
    return zm - zm[0]


run_dirs = sorted(glob.glob(os.path.join(runs_dir, "int_para2axi_cma_*")))
run_dirs = [d for d in run_dirs if os.path.isfile(os.path.join(d, "best_zernike_coeff.csv"))]
if not run_dirs:
    print(f"No completed runs with best_zernike_coeff.csv found under {runs_dir}/")
    sys.exit(1)
print(f"Found {len(run_dirs)} completed runs.")

# Physics setup -- read from the first run's own saved param file (they're
# all identical except seed_offset).
param_files = [f for f in glob.glob(os.path.join(run_dirs[0], "*.txt")) if os.path.basename(f) != "time.txt"]
params = adj.initialize_variables_from_file(param_files[0])

f_targ = params["f_targ"]
ell = 0
f_init = params["f_init"]
wl = params["wl"]
k0 = 2 * np.pi / wl
fn = params["f_num"]
w0 = f_init / (2 * fn)
rmax = params["r_maxf"] * w0
nr = params["nr"]
wR = 2 / np.pi * fn * wl
L = params["L"]

dz = params["dz"]
nzf = params["nzf"]
zF = np.linspace(f_targ - dz, f_targ + L + dz, nzf)

j = params["j"]
delta = params["delta"]
b = params.get("b", params.get("lam", 1.0))
power = 40

r, k, T_R2K, T_K2R = adj.DHTsetup(nr, rmax, ell)
prop_phase = np.exp(-1j * k**2 / (2 * k0) * zF[:, np.newaxis])
phi_I = np.exp(-((r / w0) ** power))
rho = r / rmax
Z_fit = adj.calc_Z(rho, j)
wt = adj.weight(r, wR, zF, f_targ - dz, f_targ + L + dz, delta)

zm_t = create_axiparabola(r, w0, L, f_targ)
ftf_t = adj.forward_propagation(k, phi_I, T_R2K, T_K2R, zF, zm_t, k0)
It = adj.calc_intensity(ftf_t)
maxIt = np.max(It)
It = It / maxIt


def evaluate(a_n):
    zm_r = generate_optic(a_n, Z_fit)
    ftf = adj.forward_propagation(k, phi_I, T_R2K, T_K2R, zF, zm_r, k0, phase=prop_phase)
    Ir = adj.calc_intensity(ftf) / maxIt
    Irsq, IrIt, Itsq = adj.int_pairs(r, zF, wt, Ir, It)
    fom = adj.FOM(Irsq, IrIt, Itsq)
    mu = adj.calc_mu(IrIt, Itsq)
    return float(fom / Itsq + b * abs(1 - mu)), float(mu)


costs = []
mus = []
for run_dir in run_dirs:
    a_n = np.genfromtxt(os.path.join(run_dir, "best_zernike_coeff.csv"), delimiter=",")
    cost, mu = evaluate(a_n)
    costs.append(cost)
    mus.append(mu)
    print(f"  {os.path.basename(run_dir)}: cost={cost:.6e}  mu={mu:.6f}")

costs = np.array(costs)
mus = np.array(mus)
log_costs = np.log10(np.clip(costs, 1e-15, None))
order = np.argsort(log_costs)
sorted_log = log_costs[order]

# Auto-cluster: split at the single largest gap in sorted log10(cost),
# rather than assuming a fixed threshold -- robust to a 100-seed sample
# revealing a basin we haven't seen in the smaller batches so far.
gaps = np.diff(sorted_log)
split_i = np.argmax(gaps)
split_value = (sorted_log[split_i] + sorted_log[split_i + 1]) / 2
print(f"\nLargest gap in log10(cost): {gaps[split_i]:.2f} decades, "
      f"splitting at log10(cost)={split_value:.2f} (cost={10**split_value:.3e})")

n_success = int(np.sum(log_costs <= split_value))
n_total = len(costs)
p_hat = n_success / n_total

# Wilson score interval (more reliable than normal approximation for
# proportions away from 0.5 or with smaller n).
z = 1.96
denom = 1 + z**2 / n_total
center = (p_hat + z**2 / (2 * n_total)) / denom
half_width = (z / denom) * np.sqrt(p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2))
lo, hi = center - half_width, center + half_width

print(f"\n{n_success}/{n_total} runs reached the target basin (cost <= {10**split_value:.3e})")
print(f"Point estimate: {p_hat:.1%}")
print(f"95% Wilson CI: [{lo:.1%}, {hi:.1%}]")

# mu should sit near 1 for a genuine shape match. If the "target basin" mu
# values cluster near 0 instead, this cost function has probably converged
# on the Irsq->0 degenerate exploit (mirror reflects ~nothing into the
# weighted region, which also trivially drives FOM/Itsq near zero) rather
# than a real match -- b*|1-mu| exists specifically to rule this out, so
# this is the check that matters most when b=0.
success_mus = mus[log_costs <= split_value]
print(f"\nmu among target-basin runs: min={success_mus.min():.4f}  "
      f"max={success_mus.max():.4f}  mean={success_mus.mean():.4f}")
if success_mus.mean() < 0.5:
    print("WARNING: target-basin mu is far from 1 -- this looks like the Irsq->0 "
          "degenerate exploit, not a genuine shape match.")

n_clusters_found = 1 + int(np.sum(gaps > 1.0))  # gaps > 1 decade treated as distinct clusters
if n_clusters_found > 2:
    print(f"\nNote: {n_clusters_found} distinct cost clusters detected (gap > 1 decade) -- "
          f"more than the 2 basins seen in earlier smaller batches. Worth inspecting "
          f"the per-run costs above rather than trusting the binary split alone.")
