#!/usr/bin/env python3
"""
plot_zernike_sa_comparison.py — Compare with_zernike vs sa for n=4,6,8,10.

4 rows x 3 columns:
  col 0 — mu over iterations (curves overlap, showing identical behaviour)
  col 1 — SA temperature schedule (declining but never consulted)
  col 2 — final I_OA vs z compared to target

Output: comparison_plots/zernike_sa_comparison.pdf

Run from: FF_Zernike/
    python plot_zernike_sa_comparison.py
"""

import os
import re
import sys
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "common"))
import adjoint_intensity_zernike as adj

METHODS = {
    "with_zernike": "with_zernike/int_para2axi_n/runs",
    "sa":           "sa/int_para2axi_n/runs",
}

METHOD_ORDER  = ["with_zernike", "sa"]
METHOD_LABELS = {
    "with_zernike": "Fixed Learning Rate",
    "sa":           "Simulated Annealing",
}
METHOD_COLORS = {
    "with_zernike": "tab:blue",
    "sa":           "tab:orange",
}
METHOD_STYLES = {
    "with_zernike": "-",
    "sa":           "--",
}

OUT_DIR = "comparison_plots"
os.makedirs(OUT_DIR, exist_ok=True)


def load_params(run_dir):
    pfiles = [f for f in glob.glob(os.path.join(run_dir, "*.txt"))
              if not f.endswith("time.txt")]
    return adj.initialize_variables_from_file(pfiles[0]) if pfiles else None


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
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 12,
    "lines.linewidth": 2.5,
})

for n in sorted(runs_by_n.keys()):
    method_runs = runs_by_n[n]
    print(f"n={n}: {list(method_runs.keys())}")

    fig = plt.figure(figsize=(6.5, 5))
    gs  = gridspec.GridSpec(2, 2, height_ratios=[1, 2/3], hspace=0.45, wspace=0.35,
                            top=0.95, bottom=0.2)
    ax_mu   = fig.add_subplot(gs[0, 0])
    ax_oa   = fig.add_subplot(gs[0, 1])
    ax_temp = fig.add_subplot(gs[1, :])

    mu_endpoints = []
    final_mu = {}

    first_run = next((method_runs[m] for m in METHOD_ORDER if m in method_runs), None)
    first_params = load_params(first_run) if first_run else None
    if first_params is not None:
        z_cm, target_oa = compute_target(first_params)
        ax_oa.plot(z_cm, target_oa, "k", label="Axiparabola target")
    else:
        z_cm = target_oa = None

    for method in METHOD_ORDER:
        run_dir = method_runs.get(method)
        if run_dir is None:
            continue
        params = load_params(run_dir)
        if params is None:
            continue

        data_csv = os.path.join(run_dir, "data.csv")
        if os.path.exists(data_csv):
            _, rt, tt = np.genfromtxt(data_csv, delimiter=",", unpack=True)
            mu    = adj.calc_mu(rt, tt)
            iters = np.arange(len(mu))
            ax_mu.plot(iters, mu, label=METHOD_LABELS[method],
                       color=METHOD_COLORS[method], linestyle=METHOD_STYLES[method])
            mu_endpoints.append((iters[-1], mu[-1], METHOD_COLORS[method]))
            final_mu[method] = mu[-1]

        oa_csv = os.path.join(run_dir, "int_oa.csv")
        if os.path.exists(oa_csv):
            ir_oa = np.genfromtxt(oa_csv, delimiter=",")
            if ir_oa.ndim == 1:
                ir_oa = ir_oa[np.newaxis, :]
            ax_oa.plot(z_cm, ir_oa[-1, :], label=METHOD_LABELS[method],
                       color=METHOD_COLORS[method], linestyle=METHOD_STYLES[method])

        if method == "sa":
            temp_csv = os.path.join(run_dir, "temp_history.csv")
            if os.path.exists(temp_csv):
                T_hist  = np.genfromtxt(temp_csv)
                T_iters = np.arange(len(T_hist))
                ax_temp.plot(T_iters, T_hist,
                             color=METHOD_COLORS["sa"], linewidth=2.5)


    ax_mu.axhline(1, color="black", linestyle=":", linewidth=1.5)
    ax_mu.set_xlabel("Iteration")
    ax_mu.set_ylabel(r"$\mu$")
    ax_mu.set_title(rf"$\mu$ over iterations ($n = {n}$)")

    _mu_handles, _mu_labels = ax_mu.get_legend_handles_labels()
    mu_h_by_label = dict(zip(_mu_labels, _mu_handles))
    mu_leg_h, mu_leg_l = [], []
    for method in METHOD_ORDER:
        base = METHOD_LABELS[method]
        if base not in mu_h_by_label:
            continue
        mu_val = final_mu.get(method)
        mu_leg_h.append(mu_h_by_label[base])
        mu_leg_l.append(f"$\\mu_{{f}} = {mu_val:.2f}$" if mu_val is not None else "")
    ax_mu.legend(mu_leg_h, mu_leg_l, fontsize=10, frameon=True, framealpha=0.85)

    ax_temp.set_xlabel("Iteration")
    ax_temp.set_ylabel("Temperature $T$")
    ax_temp.set_title("SA cooling schedule", pad=8)
    ax_temp.annotate(
        "Metropolis criterion\nnever invoked\n" + r"($\Delta\mathcal{L} < 0$ always)",
        xy=(0.5, 0.55), xycoords="axes fraction",
        ha="center", va="center", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                  edgecolor="gray", alpha=0.9),
    )

    ax_oa.set_xlim(-1, 6)
    ax_oa.set_xlabel(r"$z - f_{\rm targ}$ (cm)")
    ax_oa.set_ylabel("Normalized on-axis intensity")
    ax_oa.set_title("Final $I_{OA}$ vs target")

    _oa_handles, _oa_labels = ax_oa.get_legend_handles_labels()
    oa_h_by_label = dict(zip(_oa_labels, _oa_handles))
    target_h = oa_h_by_label.get("Axiparabola target")

    method_leg_h, method_leg_l = [], []
    for method in METHOD_ORDER:
        base = METHOD_LABELS[method]
        if base in oa_h_by_label:
            method_leg_h.append(oa_h_by_label[base])
            method_leg_l.append(base)

    all_h = method_leg_h + ([target_h] if target_h is not None else [])
    all_l = method_leg_l + (["Axiparabola target"] if target_h is not None else [])
    fig.legend(all_h, all_l,
               loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=len(all_h), fontsize=9, frameon=False)


    out_path = os.path.join(OUT_DIR, f"n_{n}_zernike_sa_comparison.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")

print("Done.")
