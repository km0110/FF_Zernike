#!/usr/bin/env python3
"""
plot_method_comparison.py — Compare adam / adam_sgdr / sgdr / rop for each n value.

For each n, produces a two-panel figure:
  left  — mu over iterations (one line per method)
  right — final on-axis intensity vs z compared to target (one line per method
          + black dashed target)

Output: comparison_plots/n_<N>_comparison.pdf

Run from: FF_Zernike/
    python plot_method_comparison.py
"""

import os
import re
import sys
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# All four adjoint_intensity.py files are identical; use adam's.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "common"))
import adjoint_intensity_zernike as adj

# ── Run directories for each method ───────────────────────────────────────────
METHODS = {
    "fixed_lr":  "fixed_lr/int_para2axi/runs",
    "adam":      "adam/int_para2axi_n_f20/runs",
    "adam_sgdr": "adam_sgdr/int_para2axi_n_f20/runs",
    "sgdr":      "sgdr/int_para2axi_n_f20/runs",
    "rop":       "rop/int_para2axi_n/runs",
}

METHOD_ORDER  = ["fixed_lr", "rop", "sgdr", "adam", "adam_sgdr"]
METHOD_LABELS = {
    "fixed_lr":  "Fixed Learning Rate",
    "adam":      "Adam",
    "adam_sgdr": "Adam (SGDR)",
    "sgdr":      "SGDR",
    "rop":       "Reduce on Plateau",
}
METHOD_COLORS = {"fixed_lr": "tab:green", "adam": "tab:blue", "adam_sgdr": "tab:orange", "sgdr": "tab:purple", "rop": "tab:red"}
METHOD_WIDTHS = {"fixed_lr": 3.5, "adam": 2.5, "adam_sgdr": 2.5, "sgdr": 2.5, "rop": 2.5}

OUT_DIR = "comparison_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Helper: load param file from run directory ─────────────────────────────────
def load_params(run_dir):
    pfiles = [f for f in glob.glob(os.path.join(run_dir, "*.txt"))
              if not f.endswith("time.txt")]
    return adj.initialize_variables_from_file(pfiles[0]) if pfiles else None

# ── Helper: compute normalized target OA profile ──────────────────────────────
def compute_target(params):
    f_targ = params["f_targ"]
    f_init = params["f_init"]
    wl     = params["wl"]
    k0     = 2 * np.pi / wl
    fn     = params["f_num"]
    w0     = f_init / (2 * fn)
    rmax   = params["r_maxf"] * w0
    nr     = params["nr"]
    dz     = params["dz"]
    nzf    = params["nzf"]
    L      = params["L"]
    zF     = np.linspace(f_targ - dz, f_targ + L + dz, nzf)
    r, k, T_R2K, T_K2R = adj.DHTsetup(nr, rmax, 0)
    phi_I  = np.exp(-((r / w0) ** 40))
    zm_t   = adj.create_axiparabola(r, w0, L, f_targ)
    ftf_t  = adj.forward_propagation(k, phi_I, T_R2K, T_K2R, zF, zm_t, k0)
    It     = adj.calc_intensity(ftf_t)
    z_cm   = (zF - f_targ) * 100
    return z_cm, It[0, :] / np.max(It)

# ── Collect runs grouped by n ──────────────────────────────────────────────────
runs_by_n = {}
for method, runs_dir in METHODS.items():
    for folder in sorted(glob.glob(os.path.join(runs_dir, "*"))):
        if not os.path.isdir(folder):
            continue
        m = re.search(r"_n_(\d+)_", os.path.basename(folder))
        if not m:
            continue
        n = int(m.group(1))
        runs_by_n.setdefault(n, {})[method] = folder

# ── One figure per n ───────────────────────────────────────────────────────────
font = {"size": 20}

plt.rc("font", **font)
# plt.rcParams["figure.figsize"] = (16, 9)
plt.rcParams["lines.linewidth"] = 2.5
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    # "text.latex.preamble": r"\usepackage{amsmath}\renewcommand{\seriesdefault}{\bfdefault}\boldmath",
})

for n in sorted(runs_by_n.keys()):
    method_runs = runs_by_n[n]
    print(f"n={n}: {list(method_runs.keys())}")

    fig, (ax_mu, ax_oa) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"First {n} azimuthally symmetric Zernike polynomials", y=0.95)

    final_mu   = {}
    final_iter = {}

    # Compute and plot target first so method lines render on top
    first_run = next((method_runs[m] for m in METHOD_ORDER if m in method_runs), None)
    first_params = load_params(first_run) if first_run else None
    if first_params is not None:
        z_cm, target_oa = compute_target(first_params)
        ax_oa.plot(z_cm, target_oa, "k--", label="Axiparabola target")
    else:
        z_cm = target_oa = None

    for method in METHOD_ORDER:
        run_dir = method_runs.get(method)
        if run_dir is None:
            continue

        params = load_params(run_dir)
        if params is None:
            continue

        # ── mu over iterations ────────────────────────────────────────────────
        data_csv = os.path.join(run_dir, "data.csv")
        if os.path.exists(data_csv):
            _, rt, tt = np.genfromtxt(data_csv, delimiter=",", unpack=True)
            mu    = adj.calc_mu(rt, tt)
            iters = np.arange(len(mu))
            ax_mu.plot(iters, mu, label=METHOD_LABELS[method],
                       color=METHOD_COLORS[method], linewidth=METHOD_WIDTHS[method])
            final_mu[method]   = mu[-1]
            final_iter[method] = int(iters[-1])

        # ── final I_OA ────────────────────────────────────────────────────────
        oa_csv = os.path.join(run_dir, "int_oa.csv")
        if os.path.exists(oa_csv):
            ir_oa = np.genfromtxt(oa_csv, delimiter=",")
            if ir_oa.ndim == 1:
                ir_oa = ir_oa[np.newaxis, :]
            ax_oa.plot(z_cm, ir_oa[-1, :], label=METHOD_LABELS[method],
                       color=METHOD_COLORS[method], linewidth=METHOD_WIDTHS[method])

    ax_mu.axhline(1, color="black", linestyle=":", linewidth=1.5)

    ax_mu.set_xscale("log")
    ax_mu.set_xlabel("Iteration")
    ax_mu.set_ylabel(r"$\mu$")
    ax_mu.set_title(r"$\mu$ over iterations")

    ax_oa.set_xlim(-1, 6)
    ax_oa.set_xlabel(r"$z - f_{\rm targ}$ (cm)")
    ax_oa.set_ylabel("Normalized on-axis intensity")
    ax_oa.set_title("Final $I_{OA}$ vs target")

    # Legend inside ax_mu: final mu values only (colored lines as handles)

    _oa_handles, _oa_labels = ax_oa.get_legend_handles_labels()
    oa_h_by_label = dict(zip(_oa_labels, _oa_handles))
    target_h = oa_h_by_label.get("Axiparabola target")

    def leg_label(method):
        base   = METHOD_LABELS[method]
        mu_val = final_mu.get(method)
        it     = final_iter.get(method)
        return (f"{base} ($\\mu_{{f}} = {mu_val:.2f}$, iter {it:,})"
                if mu_val is not None else base)

    def leg_handle(method):
        return oa_h_by_label.get(METHOD_LABELS[method])

    # column-major fill with ncol=3:
    # col 0: fixed_lr (row 0), adam (row 1)
    # col 1: rop      (row 0), adam_sgdr (row 1)
    # col 2: sgdr     (row 0), axiparabola (row 1)
    all_h = [leg_handle("fixed_lr"), leg_handle("adam"),
             leg_handle("rop"),      leg_handle("adam_sgdr"),
             leg_handle("sgdr"),     target_h]
    all_l = [leg_label("fixed_lr"), leg_label("adam"),
             leg_label("rop"),      leg_label("adam_sgdr"),
             leg_label("sgdr"),     "Axiparabola target"]

    fig.tight_layout(rect=[0, 0, 1, 0.988])

    fig.legend(all_h, all_l,
               loc="lower center", bbox_to_anchor=(0.5, 0),
               ncol=3, fontsize=15, frameon=False)

    fig.subplots_adjust(bottom=0.25)

    out_path = os.path.join(OUT_DIR, f"n_{n}_comparison.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")

print("Done.")
