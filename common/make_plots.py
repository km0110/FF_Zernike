#!/usr/bin/env python3
"""
make_plots.py — Generate plots from a completed simulation run.

Shared across all experiment directories (adam, adam_newfom, adam_sgdr,
cma_es, rop, sgdr, fixed_lr, without_zernike) -- this file lives outside all
of them in common/.

Which FOM/gradient module to use is genuinely different per directory
(Zernike-standard, Zernike+b-penalty "newfom", or without_zernike's direct-pixel
parameterization) and must be given explicitly via --fom-module -- there is
no name-based auto-detection here on purpose (a same-named local file used to
shadow the wrong module silently; see git history).

Usage:
    python make_plots.py <path to run directory> --fom-module {zernike,zernike_newfom,local} [--generations]

--fom-module zernike          adjoint_intensity_zernike.py (in this directory) -- adam, adam_sgdr, cma_es, rop, sgdr, fixed_lr
--fom-module zernike_newfom   adjoint_intensity_newfom.py (in this directory) -- adam_newfom
--fom-module local            this run's own directory's adjoint_intensity.py -- without_zernike only

--generations labels plot axes/titles "Generation" instead of "Iteration"
(for CMA-ES-style runs).

Toggle plots on/off in the PLOTS dict below. Only the data files
required by the enabled plots will be loaded.
"""

import sys
import os
import glob
import time
import inspect
import argparse
import numpy as np
from datetime import timedelta

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("path")
parser.add_argument("--fom-module", choices=["zernike", "zernike_newfom", "local"], default="zernike")
parser.add_argument("--generations", action="store_true")
args = parser.parse_args()

if args.fom_module == "local":
    # without_zernike's own real (non-shared) adjoint_intensity.py.
    sys.path.insert(0, os.getcwd())
    import adjoint_intensity as adj
elif args.fom_module == "zernike_newfom":
    import adjoint_intensity_newfom as adj
else:
    import adjoint_intensity_zernike as adj
import plot as plots

path = args.path.rstrip("/")
step_label = "Generation" if args.generations else "Iteration"

if not os.path.isdir(path):
    print(f"Directory not found: {path}")
    sys.exit(1)

# ── Plot selection ─────────────────────────────────────────────────────────────
PLOTS = {
    "fom_mu":                False,
    "percent_change_mu":     False,
    "total_change":          False,
    "gradient":              False,
    "mirror_comp":           False,
    "mirror_comp_shifted":   False,
    "I_OA_over_iterations":  False,
    "I_f_over_iterations":   True,
    "I_OA_frames":           False,
    "step0_intensity":       False,
    "zernikes":              False,
    # non-int only:
    "phi_f_over_iterations":  False,
    "phi_OA_over_iterations": False,
    "step0_phi":              False,
}

# ── Data file → variable dependencies ─────────────────────────────────────────
DATA_DEPS = {
    "fom_mu":                ["data"],
    "percent_change_mu":     ["data"],
    "total_change":          ["gradient"],
    "gradient":              ["gradient"],
    "mirror_comp":           ["mirror"],
    "mirror_comp_shifted":   ["mirror"],
    "I_OA_over_iterations":  ["int_oa"],
    "I_f_over_iterations":   ["int_f"],
    "I_OA_frames":           ["int_oa"],
    "step0_intensity":       ["int_oa", "int_f"],
    "zernikes":              ["zernike_coeff", "zernike_grad"],
    "phi_f_over_iterations":  ["phi_r_focus"],
    "phi_OA_over_iterations": ["phi_r_OA"],
    "step0_phi":              ["phi_r_OA", "phi_r_focus"],
    # per-step (kept for reference, disabled above):
    # "psi_and_source":       ["psi", "source"],
}

needed = set()
for name, enabled in PLOTS.items():
    if enabled:
        needed.update(DATA_DEPS.get(name, []))

# ── Locate parameter file ──────────────────────────────────────────────────────
param_files = [
    f for f in glob.glob(os.path.join(path, "*.txt")) if not f.endswith("time.txt")
]
if not param_files:
    print(f"No parameter file found in {path}")
    sys.exit(1)
param_file = param_files[0]

# ── Reconstruct parameters and derived quantities ──────────────────────────────
params = adj.initialize_variables_from_file(param_file)

f_targ = params["f_targ"]
f_init = params["f_init"]
ell = 0
wl = params["wl"]
k0 = 2 * np.pi / wl
fn = params["f_num"]
w0 = f_init / (2 * fn)
rmax = params["r_maxf"] * w0
nr = params["nr"]

dz = params["dz"]
nzf = params["nzf"]
L = params.get("L", 0)
print(L)
# para2para scripts don't include L in zF (point focus target); all others do
zF_L = 0 if "2para" in os.path.basename(path) else L
zF = np.linspace(f_targ - dz, f_targ + zF_L + dz, nzf)
fl = len(zF) // 2

lr = params.get("lr")  # gradient-free optimizers (e.g. CMA-ES) have no lr; only total_change needs it
skip = params["skip"]
steps_param = params["steps"]

[r, k, T_R2K, T_K2R] = adj.DHTsetup(nr, rmax, ell)

power = 40 if 'axi' in path else 2
phi_I = np.exp(-((r / w0) ** power))
zm_t = adj.create_axiparabola(r, w0, L, f_targ) if 'axi' in path else adj.create_starting_optic(r, 2 * f_targ)
ftf_t = adj.forward_propagation(k, phi_I, T_R2K, T_K2R, zF, zm_t, k0)
It = adj.calc_intensity(ftf_t)

is_int = 'int' in os.path.basename(path)
if is_int:
    # main_int_*.py normalizes It (and saves Ir already divided by this same
    # maxIt) -- field-matching runs never normalize It at all, so this must
    # stay conditional on is_int to match what was actually saved.
    It = It / np.max(It)

# ── Load only required data files ─────────────────────────────────────────────
def _load(filename):
    return np.genfromtxt(os.path.join(path, filename), delimiter=",")

gr     = _load("gradient.csv")    if "gradient"    in needed else None
mr     = _load("mirror.csv")      if "mirror"      in needed else None
ir_oa  = _load("int_oa.csv")      if "int_oa"      in needed else None
ir_f   = _load("int_f.csv")       if "int_f"       in needed else None
src    = _load("source.csv")      if "source"      in needed else None
psis   = _load("psi.csv")         if "psi"         in needed else None
phr_oa = _load("phi_r_OA.csv")    if "phi_r_OA"    in needed else None
phr_f  = _load("phi_r_focus.csv") if "phi_r_focus" in needed else None
zern = _load("zernike_coeff.csv") if "zernike_coeff" in needed else None
# zernike_grad.csv is only written by some Zernike-coefficient-parameterized
# pipelines; tolerate it being absent rather than crashing the whole plotting run.
try:
    zern_grad = _load("zernike_grad.csv") if "zernike_grad" in needed else None
except OSError:
    zern_grad = None

if "data" in needed:
    rr, rt, tt = np.genfromtxt(os.path.join(path, "data.csv"), delimiter=",", unpack=True)
    # adam_newfom's adjoint_intensity.FOM takes an extra b weighting term;
    # other directories' FOM doesn't accept it. Falls back to "alpha" for
    # old saved runs from before the alpha -> b rename.
    b = params.get("b", params.get("alpha", 1.0))
    fom_kwargs = {"b": b} if "b" in inspect.signature(adj.FOM).parameters else {}
    figom  = adj.FOM(rr, rt, tt, **fom_kwargs)
    mu_val = adj.calc_mu(rt, tt)
else:
    figom = mu_val = None

# Infer actual save interval from data: if every step was saved the row count
# equals steps_param, so multiplying label indices by skip would be wrong.
_ref = next((a for a in (ir_oa, phr_oa, ir_f, phr_f) if a is not None), None)
plot_skip = 1 if (_ref is not None and len(_ref) == steps_param) else skip

# ── Ensure per-iteration subdirectories exist ──────────────────────────────────
subdirs = ["I_OA", "I_f", "g", "mirror", "psi_S"]
if not is_int:
    subdirs += ["pr_f", "pr_oa"]
for subdir in subdirs:
    os.makedirs(os.path.join(path, subdir), exist_ok=True)

# ── Generate plots ─────────────────────────────────────────────────────────────
print("Generating plots...")
pstart = time.time()

n_steps = len(gr) if gr is not None else None

if PLOTS["fom_mu"]:
    plots.plot_fom_mu(path, figom, mu_val, step_label=step_label)
if PLOTS["percent_change_mu"]:
    plots.plot_percent_change_mu(path, mu_val, step_label=step_label)
if PLOTS["total_change"]:
    plots.plot_total_change(path, r, w0, gr, lr, wl, n_steps)
if PLOTS["gradient"]:
    plots.plot_gradient(path, gr, r, w0, step_label=step_label)
if PLOTS["mirror_comp"]:
    plots.plot_mirror_comp(path, r, w0, zm_t, mr, do_shift=False)
if PLOTS["mirror_comp_shifted"]:
    plots.plot_mirror_comp(path, r, w0, zm_t, mr, do_shift=True)
oa_xlim = (-2, L * 100 + 2) if "para2axi" in os.path.basename(path) else None
if PLOTS["I_OA_over_iterations"]:
    plots.plot_I_OA_over_iterations(path, ir_oa, zF, f_targ, It[0, :], plot_skip,
                                    xlim=oa_xlim, step_label=step_label)
    if oa_xlim is not None:
        plots.plot_I_OA_over_iterations_zoom(path, ir_oa, zF, f_targ, It[0, :], plot_skip,
                                             xlim=oa_xlim, step_label=step_label)
if PLOTS["I_OA_frames"]:
    plots.plot_I_OA(path, ir_oa, zF, f_targ, It[0, :], plot_skip)
if PLOTS["I_f_over_iterations"]:
    plots.plot_I_f_over_iterations(path, ir_f, r, w0, It[:, fl], plot_skip, step_label=step_label)
if PLOTS["step0_intensity"]:
    plots.plot_step0_intensity(path, ir_oa, ir_f, zF, f_targ, r, w0, It, fl)
if PLOTS["zernikes"]:
    plots.plot_zernikes(path, zern, zern_grad=zern_grad, step_label=step_label)
if not is_int:
    if PLOTS["phi_f_over_iterations"]:
        plots.plot_phi_f_over_iterations(path, phr_f, r, w0, np.real(ftf_t), fl, plot_skip, step_label=step_label)
    if PLOTS["phi_OA_over_iterations"]:
        plots.plot_phi_OA_over_iterations(path, phr_oa, zF, f_targ, np.real(ftf_t), plot_skip,
                                          xlim=oa_xlim, step_label=step_label)
    if PLOTS["step0_phi"]:
        plots.plot_step0_phi(path, phr_oa, phr_f, zF, f_targ, r, w0, np.real(ftf_t), fl)
# plots.plot_psi_and_source(path, psis, r, w0, src, zF, f_targ, skip)

pend = time.time()
elapsed = str(timedelta(seconds=pend - pstart))
print(f"Done. Plotting time: {elapsed}")

with open(os.path.join(path, "time.txt"), "a") as f:
    f.write(f"\nPlotting time: {elapsed}")
