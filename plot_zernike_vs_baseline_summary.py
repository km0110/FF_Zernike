#!/usr/bin/env python3
"""
plot_zernike_vs_baseline_summary.py — without_zernike vs adam(+Zernike), across
all four test cases: para2para, para2axi, int_para2para, int_para2axi.

Apples-to-apples run selection: n=3 Zernike modes for the para2para-family
cases, n=4 for the para2axi-family cases (matching mode counts available on
both sides), with f_init/tol matched to the corresponding without_zernike run.

Output:
  comparison_plots/zernike_axi_family_grid.pdf  — para2axi / int_para2axi:
                                                   mu convergence (top row) +
                                                   final I_OA vs target (bottom)
  comparison_plots/zernike_para_family_grid.pdf — para2para / int_para2para:
                                                   mu convergence (top row) +
                                                   final I_OA vs target (bottom)
  comparison_plots/zernike_axi_mirror_comparison.pdf  — para2axi / int_para2axi:
                                                         |target - final| mirror
  comparison_plots/zernike_para_mirror_comparison.pdf — para2para / int_para2para:
                                                         |target - final| mirror
                                                         (field-based cases get
                                                         spike-unwrapped first)

Run from: FF_Zernike/
    python plot_zernike_vs_baseline_summary.py
"""

import os
import sys
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "common"))
import adjoint_intensity_zernike as adj
from plot import spikes, shift

OUT_DIR = "comparison_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Run directories (apples-to-apples: n=3 for para2para family, n=4 for
#    para2axi family; f_init/tol matched to the without_zernike counterpart) ──
TEST_CASES = {
    "para2para": {
        "kind": "focal",
        "title": "Field-based Parabola $\\to$ Parabola",
        "without_zernike": "without_zernike/para_to_para/runs/para_delta_lr_1e-13_f_targ_0.05_f_init_0.045_nr_1000_r_maxf_5_nzf_501_dz_0.02_steps_5001_f_num_25_ell_0_wl_1e-06_conic_-1_delta_1_tol_1e-14_sg_0_order_0_skip_100_temp_10_factor_0.5",
        "adam": "adam/para2para/runs/para2para_adam_d_lr_1e-08_ft_0.05_fi_0.045_n_3_nr_1000_rmx_5_nzf_1001_dz_0.02_L_0.05_st_10000_sk_1_fn_25_eps1e-6_tol_1e-11",
    },
    "para2axi": {
        "kind": "axial",
        "title": "Field-based Parabola $\\to$ Axiparabola",
        "without_zernike": "without_zernike/para_to_axi/runs/para2axi_delta_lr_1e-13_f_targ_0.5_f_init_0.5_nr_2000_r_maxf_1.3_nzf_1001_dz_0.05_L_0.05_steps_10001_f_num_20_ell_0_wl_1e-06_conic_-1_delta_1_tol_1e-14_sg_0_order_0_skip_100_temp_10_factor_0.5",
        "adam": "adam/para2axi/runs/para2axi_adam_d_lr_1e-08_ft_0.5_fi_0.5_n_4_nr_2001_rmx_1.3_nzf_1001_dz_0.05_L_0.05_st_25000_sk_1_fn_20_eps1e-6_tol_1e-11",
    },
    "int_para2para": {
        "kind": "focal",
        "title": "Intensity-based Parabola $\\to$ Parabola",
        "without_zernike": "without_zernike/int_para_to_para/runs/int_para2para_d_lr_1e-10_ft_0.05_fi_0.045_nr_1000_rmx_5_nzf_501_dz_0.02_L_0_st_1000_sk_1_fn_25",
        "adam": "adam/int_para2para/runs/int_para2para_adam_d_lr_1e-08_ft_0.05_fi_0.045_n_3_nr_1001_rmx_5_nzf_501_dz_0.02_L_0_st_10000_sk_1_fn_25_eps_1e-6_tol_1e-11",
    },
    "int_para2axi": {
        "kind": "axial",
        "title": "Intensity-based Parabola $\\to$ Axiparabola",
        "without_zernike": "without_zernike/int_para_to_axi/runs/int_para2axi_delta_lr_1e-13_f_targ_0.5_f_init_0.5_nr_2000_r_maxf_1.3_nzf_1001_dz_0.05_L_0.05_steps_10001_f_num_20_ell_0_wl_1e-06_conic_-1_delta_1_tol_1e-14_sg_0_order_0_skip_100_temp_10_factor_0.5",
        "adam": "adam/int_para2axi_n_f20_fi_0.5/runs/int_para2axi_adam_d_lr_1e-08_ft_0.5_fi_0.5_n_4_nr_2001_rmx_1.3_nzf_1001_dz_0.05_L_0.05_st_12601_sk_10_fn_20_eps1e-6_tol_1e-11",
    },
}
TEST_ORDER = ["para2para", "para2axi", "int_para2para", "int_para2axi"]

METHOD_ORDER  = ["without_zernike", "adam"]
METHOD_LABELS = {"without_zernike": "Point-to-Point", "adam": "Zernike + Adam"}
METHOD_COLORS = {"without_zernike": "tab:gray", "adam": "tab:blue"}
METHOD_WIDTHS = {"without_zernike": 3.0, "adam": 2.5}


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_params(run_dir):
    pfiles = [f for f in glob.glob(os.path.join(run_dir, "*.txt"))
              if not f.endswith("time.txt")]
    return adj.initialize_variables_from_file(pfiles[0]) if pfiles else None


def compute_target(params, kind):
    """Return (x, target_normalized, target_peak, r, w0, zm_t) on this run's own grid.

    The reported profile is always the on-axis intensity I_OA vs
    x = (z - f_targ) in cm. `kind` only selects the mirror physics:
    "focal" -> point-focus mirror, narrow zF window (+-dz around f_targ).
    "axial" -> axiparabola mirror, zF window extended by L (flying focus).
    r, w0, zm_t are the radial grid, beam waist, and target mirror height
    profile (metres), for the mirror-shape comparison.
    """
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
    L      = params.get("L", 0)

    zF_L = 0 if kind == "focal" else L
    zF   = np.linspace(f_targ - dz, f_targ + zF_L + dz, nzf)
    r, k, T_R2K, T_K2R = adj.DHTsetup(nr, rmax, 0)

    if kind == "focal":
        phi_I = np.exp(-((r / w0) ** 2))
        zm_t  = adj.create_starting_optic(r, 2 * f_targ)
    else:
        phi_I = np.exp(-((r / w0) ** 40))
        zm_t  = adj.create_axiparabola(r, w0, L, f_targ)

    ftf_t = adj.forward_propagation(k, phi_I, T_R2K, T_K2R, zF, zm_t, k0)
    It    = adj.calc_intensity(ftf_t)

    x   = (zF - f_targ) * 100
    raw = It[0, :]

    peak = np.max(raw)
    return x, raw / peak, peak, r, w0, zm_t


# ── Family figures: para2para-family and para2axi-family, both I_OA ────────
font = {"size": 16}
plt.rc("font", **font)
plt.rcParams["lines.linewidth"] = 2.2
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

FAMILIES = [
    {
        "cases":        ["para2axi", "int_para2axi"],
        "kind":         "axial",
        "quantity":     r"I_{OA}",
        "profile_xlabel": r"$z - f_{\rm targ}$ (cm)",
        "profile_xlim": (-1, 6),
        "out_name":     "zernike_axi_family_grid.pdf",
    },
    {
        "cases":        ["para2para", "int_para2para"],
        "kind":         "focal",
        "quantity":     r"I_{OA}",
        "profile_xlabel": r"$z - f_{\rm targ}$ (cm)",
        "profile_xlim": (-2, 2),
        "out_name":     "zernike_para_family_grid.pdf",
    },
]

for family in FAMILIES:
    cases     = family["cases"]
    kind      = family["kind"]
    quantity  = family["quantity"]
    out_name  = family["out_name"]

    fig, axes = plt.subplots(2, len(cases), figsize=(6 * len(cases), 10), squeeze=False)

    for col, tc in enumerate(cases):
        info   = TEST_CASES[tc]
        is_int = tc.startswith("int_")
        ax_mu, ax_prof = axes[0, col], axes[1, col]

        target_plotted = False

        for method in METHOD_ORDER:
            run_dir = info[method]
            params  = load_params(run_dir)
            if params is None:
                continue

            x, target, peak, _, _, _ = compute_target(params, kind)
            if not target_plotted:
                ax_prof.plot(x, target, "k--", label=f"Target ${quantity}$", linewidth=2.0)
                target_plotted = True

            data_csv = os.path.join(run_dir, "data.csv")
            if os.path.exists(data_csv):
                _, rt, tt = np.genfromtxt(data_csv, delimiter=",", unpack=True)
                mu    = adj.calc_mu(rt, tt)
                iters = np.arange(len(mu))
                mu_label = f"$\\mu_f = {mu[-1]:.3f}$"
                ax_mu.plot(iters, mu, label=mu_label,
                           color=METHOD_COLORS[method], linewidth=METHOD_WIDTHS[method])

            profile_csv = os.path.join(run_dir, "int_oa.csv")
            if os.path.exists(profile_csv):
                ir = np.genfromtxt(profile_csv, delimiter=",")
                if ir.ndim == 1:
                    ir = ir[np.newaxis, :]
                ir_scale = 1.0 if is_int else peak
                ax_prof.plot(x, ir[-1, :] / ir_scale, label=METHOD_LABELS[method],
                             color=METHOD_COLORS[method], linewidth=METHOD_WIDTHS[method])

        ax_mu.axhline(1, color="black", linestyle=":", linewidth=1.2)
        ax_mu.set_xscale("log")
        ax_mu.set_xlabel("Iteration")
        ax_mu.set_title(info["title"])
        ax_mu.legend(fontsize=11, loc="lower right")
        if col == 0:
            ax_mu.set_ylabel(r"$\mu$")

        ax_prof.set_xlim(*family["profile_xlim"])
        ax_prof.set_xlabel(family["profile_xlabel"])
        ax_prof.set_ylabel(f"Normalized ${quantity}$")
        ax_prof.set_title(f"Final ${quantity}$ vs Target")

    prof_handles, prof_labels = axes[1, 0].get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.legend(prof_handles, prof_labels, loc="lower center", ncol=3, frameon=False,
               fontsize=15, bbox_to_anchor=(0.5, 0))

    out_path = os.path.join(OUT_DIR, out_name)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

# ── Figures: |target mirror - final mirror|, para family / axi family ──────
# Field-based (non-int) runs optimize phase, so the raw final mirror can carry
# half-wavelength spike discontinuities; unwrap those via spikes()/shift()
# before differencing against the target. Intensity-based runs don't need this.
MIRROR_FAMILIES = [
    {"cases": ["para2axi", "int_para2axi"], "out_name": "zernike_axi_mirror_comparison.pdf", "xlim": (0, 1.3)},
    {"cases": ["para2para", "int_para2para"], "out_name": "zernike_para_mirror_comparison.pdf", "xlim": (0, 2)},
]

for mfamily in MIRROR_FAMILIES:
    cases    = mfamily["cases"]
    out_name = mfamily["out_name"]
    xlim     = mfamily["xlim"]

    fig3, axes3 = plt.subplots(1, len(cases), figsize=(6 * len(cases), 6), squeeze=False)
    axes3 = axes3[0]

    for col, tc in enumerate(cases):
        info   = TEST_CASES[tc]
        kind   = info["kind"]
        is_int = tc.startswith("int_")
        ax     = axes3[col]

        for method in METHOD_ORDER:
            run_dir = info[method]
            params  = load_params(run_dir)
            if params is None:
                continue

            _, _, _, r, w0, zm_t = compute_target(params, kind)

            mirror_csv = os.path.join(run_dir, "mirror.csv")
            if os.path.exists(mirror_csv):
                mr = np.genfromtxt(mirror_csv, delimiter=",")
                if mr.ndim == 1:
                    mr = mr[np.newaxis, :]
                final_mirror = mr[-1, :]
                if not is_int:
                    final_mirror = shift(final_mirror, spikes(final_mirror))
                diff = np.abs(zm_t - final_mirror)
                ax.plot(r / w0, diff, label=METHOD_LABELS[method],
                        color=METHOD_COLORS[method], linewidth=METHOD_WIDTHS[method])

        ax.set_yscale("log")
        ax.set_xlim(*xlim)
        ax.set_xlabel(r"$r / w_0$")
        if col == 0:
            ax.set_ylabel(r"$|$Target $-$ Final Mirror$|$ (m)")
        ax.set_title(info["title"])

    mirror_handles, mirror_labels = axes3[0].get_legend_handles_labels()
    fig3.tight_layout(rect=[0, 0.1, 1, 1])
    fig3.legend(mirror_handles, mirror_labels, loc="lower center", ncol=3, frameon=False,
                fontsize=15, bbox_to_anchor=(0.5, 0))

    mirror_path = os.path.join(OUT_DIR, out_name)
    fig3.savefig(mirror_path, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved {mirror_path}")

print("Done.")
