#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os

import matplotlib as mpl
# This module only ever calls savefig(), never plt.show() -- and several
# drivers (e.g. CMA-ES restarts) fork worker processes, where the macOS
# GUI backend is not fork-safe. Agg has no GUI/toolkit state to corrupt.
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np

# from adjoint import create_initial_coeff, calc_Z

font = {"size": 30}

plt.rc("font", **font)
plt.rcParams["figure.figsize"] = (16, 9)
plt.rcParams["lines.linewidth"] = 4
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    # "text.latex.preamble": r"\usepackage{amsmath}\renewcommand{\seriesdefault}{\bfdefault}\boldmath",
})


def plot_total_change(path, r, w0, gr, lr, wl, steps, xlim=(0, 2), spikes=None, fontsize=30):
    """Plot the cumulative mirror deformation summed over all adjoint steps.

    Saves the figure to ``<path>/total_change.pdf``.

    Parameters
    ----------
    path : str
        Output directory where the PDF will be saved.
    r : np.ndarray, shape (N,)
        Radial coordinate array (metres).
    w0 : float
        Beam waist radius used to normalise the radial axis (metres).
    gr : np.ndarray, shape (steps, N)
        Gradient history; each row is the mirror gradient at one adjoint step.
    lr : float
        Learning rate (step size) applied to the gradient update.
    wl : float
        Wavelength used to normalise the deformation to units of lambda (metres).
    steps : int
        Total number of optimisation steps, used for the legend label.
    xlim : tuple of float, optional
        ``(xmin, xmax)`` in units of ``r/w0``. Applied only when
        ``max(r/w0) > 2``. Default is ``(0, 2)``.
    spikes : array-like of int or None, optional
        Indices into ``r`` at which to draw vertical dashed lines marking
        discontinuities. Default is ``None`` (no lines drawn).
    fontsize : int, optional
        Base font size for all text elements. Default is 30.

    Returns
    -------
    None
        Figure is saved to disk and the pyplot state is closed.
    """
    with plt.rc_context({"font.size": fontsize}):
        norm_r = r / w0
        plt.scatter(
            norm_r, -1 * np.sum(gr, axis=0) * lr / wl, label="After %d Steps" % (steps)
        )
        plt.title("Total change in the mirror")
        plt.xlabel(r"$r/w_0$")
        plt.ylabel(r"$\sum\delta z/\lambda$")
        plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
        plt.axhline(0, c="k", linewidth=1)
        if max(norm_r) > 2:
            plt.xlim(xlim)
        else:
            plt.xlim(xmin=0)
        if spikes != None:
            for s in r[spikes]:
                plt.axvline(s / w0, linestyle="--", c="k", linewidth=2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path + "/total_change.pdf")
        plt.close()


# In[3]:
def plot_fom_mu(path, fom, mu, fontsize=30, step_label="Iteration"):
    """Plot the figure of merit and step-size parameter mu on shared axes.

    The figure of merit is normalised to its initial value so both curves
    share a comparable scale. Saves to ``<path>/fom_mu.pdf``.

    Parameters
    ----------
    path : str
        Output directory where the PDF will be saved.
    fom : np.ndarray, shape (iterations,)
        Figure-of-merit value recorded at each optimisation iteration.
    mu : np.ndarray, shape (iterations,)
        Step-size (learning-rate) parameter recorded at each iteration.
    fontsize : int, optional
        Base font size for all text elements. Default is 30.

    Returns
    -------
    None
        Figure is saved to disk and the pyplot state is closed.
    """
    with plt.rc_context({"font.size": fontsize}):
        fig, ax1 = plt.subplots()

        color = "tab:red"
        ax1.set_xlabel(step_label + "s")
        ax1.set_ylabel(r"$\mu$", color=color)
        ax1.plot(mu, color=color)
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()
        color = "tab:blue"
        ax2.set_ylabel("Figure of Merit", color=color)
        ax2.plot(fom / fom[0], color=color)
        ax2.tick_params(axis="y", labelcolor=color)

        plt.tight_layout()
        plt.savefig(path + "/fom_mu.pdf")
        plt.close()


def plot_percent_change_mu(path, mu, fontsize=30, step_label="Iteration"):
    """Plot the iteration-over-iteration percent change of the step-size mu.

    Useful for diagnosing whether the learning rate has converged or is still
    oscillating. Saves to ``<path>/pc_mu.pdf``.

    Parameters
    ----------
    path : str
        Output directory where the PDF will be saved.
    mu : np.ndarray, shape (iterations,)
        Step-size parameter recorded at each optimisation iteration.
    fontsize : int, optional
        Base font size for all text elements. Default is 30.

    Returns
    -------
    None
        Figure is saved to disk and the pyplot state is closed.
    """
    with plt.rc_context({"font.size": fontsize}):
        percent_change = np.abs((np.diff(mu) / mu[:-1]) * 100)
        iterations = range(1, len(percent_change) + 1)
        plt.plot(iterations, percent_change)
        plt.xlabel(step_label)
        plt.ylabel(r"$|\%\ \mathrm{Change\ in}\ \mu|$")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(path + "/pc_mu.pdf")
        plt.close()


def plot_integrals(path, prsq, prpt, ptsq, fontsize=30, step_label="Iteration"):
    """Plot the overlap integrals used in the figure-of-merit decomposition.

    Compares ``<phi_R^2><phi_T^2>`` against ``<phi_R phi_T>^2`` over
    iterations to illustrate how the reflected field overlaps with the target.
    Saves to ``<path>/prpt.pdf``.

    Parameters
    ----------
    path : str
        Output directory where the PDF will be saved.
    prsq : np.ndarray, shape (iterations,)
        Integral of ``|phi_R|^2`` at each iteration.
    prpt : np.ndarray, shape (iterations,)
        Cross-integral ``<phi_R phi_T>`` at each iteration.
    ptsq : np.ndarray, shape (iterations,)
        Integral of ``|phi_T|^2`` at each iteration.
    fontsize : int, optional
        Base font size for all text elements. Default is 30.

    Returns
    -------
    None
        Figure is saved to disk and the pyplot state is closed.
    """
    with plt.rc_context({"font.size": fontsize}):
        plt.plot(prsq * ptsq, label=r"$\langle\phi_R^2\rangle \langle\phi_T^2\rangle$")
        plt.plot(prpt * prpt, label=r"$\langle\phi_R\phi_T\rangle^2$")
        plt.xlabel(step_label + "s")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path + "/prpt.pdf")
        plt.close()


def spikes(mirror):
    """Detect abrupt downward discontinuities in the mirror profile.

    Uses the second discrete difference to find locations where the curvature
    drops sharply below a fixed threshold, indicating numerical spikes
    introduced by the optimiser.

    Parameters
    ----------
    mirror : np.ndarray, shape (N,)
        Mirror surface height profile (metres).

    Returns
    -------
    np.ndarray of int
        Indices into ``mirror`` where spikes are detected.
    """
    ddfmr = np.diff(mirror, n=2)
    return np.where(ddfmr < -4.5e-7)[0] + 1


def shift(mirror, spikes):
    """Remove cumulative vertical offsets introduced by mirror spikes.

    Each segment between consecutive spike indices is shifted downward by an
    increasing multiple of ``5e-7`` metres so that the segments lie on a
    common baseline and the spike artefacts become visually apparent.

    Parameters
    ----------
    mirror : np.ndarray, shape (N,)
        Mirror surface height profile to be corrected (metres).
    spikes : array-like of int
        Indices into ``mirror`` marking the start of each spiked segment,
        as returned by :func:`spikes`.

    Returns
    -------
    np.ndarray, shape (N,)
        Copy of ``mirror`` with each post-spike segment shifted down by
        ``(segment_index + 1) * 5e-7`` metres.
    """
    new_mir = np.copy(mirror)
    n = len(spikes)
    spikes = np.append(spikes, len(mirror))
    for i in range(n):
        start = spikes[i]
        end = spikes[i + 1]
        new_mir[start:end] -= (i + 1) * 5e-7
    return new_mir


def plot_mirror_comp(path, r, w0, z_m, mr, steps=None, xlim=(0, 2), ylim=(0, 2e-5), do_shift=False, fontsize=30):
    """Compare the optimised mirror profile to the initial and target profiles.

    When ``do_shift=True`` a two-panel figure is produced showing the raw
    (unshifted) final mirror alongside a spike-corrected (shifted) version.
    Otherwise a single panel is saved. Saves to ``<path>/mirror_comp.pdf`` or
    ``<path>/mirror_comp_shifted.pdf``.

    Parameters
    ----------
    path : str
        Output directory where the PDF will be saved.
    r : np.ndarray, shape (N,)
        Radial coordinate array (metres).
    w0 : float
        Beam waist radius used to normalise the radial axis (metres).
    z_m : np.ndarray, shape (N,)
        Target mirror surface height profile (metres).
    mr : np.ndarray, shape (iterations, N)
        Mirror profile recorded at every optimisation iteration; ``mr[0]`` is
        the initial profile and ``mr[-1]`` is the final profile.
    steps : int or None, optional
        Number of optimisation steps used for the legend label.
        If ``None``, the label reads ``"Final Mirror"``. Default is ``None``.
    xlim : tuple of float, optional
        ``(xmin, xmax)`` in units of ``r/w0``. Applied only when
        ``max(r/w0) > 2``. Default is ``(0, 2)``.
    ylim : tuple of float, optional
        ``(ymin, ymax)`` in metres. Applied only when ``max(r/w0) > 2``.
        Default is ``(0, 2e-5)``.
    do_shift : bool, optional
        If ``True``, produce a two-panel figure with unshifted and
        spike-corrected panels. Default is ``False``.
    fontsize : int, optional
        Base font size for all text elements. Default is 30.

    Returns
    -------
    None
        Figure is saved to disk and the pyplot state is closed.
    """
    with plt.rc_context({"font.size": fontsize}):
        norm_r = r / w0
        sp = spikes(mr[-1])
        label = "After %d Steps" % steps if steps is not None else "Final Mirror"

        if do_shift:
            final_unshifted = mr[-1]
            final_shifted = shift(mr[-1], sp)

            _, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(16, 18), sharex=True)

            # --- top: unshifted ---
            ax_top.plot(norm_r, z_m, label="Target Mirror")
            ax_top.plot(norm_r, mr[0], label="Initial Mirror")
            ax_top.plot(norm_r, final_unshifted, c='k',linestyle="--", linewidth=5, label=label)
            for s in r[sp]:
                ax_top.axvline(s / w0, alpha=0.5, linestyle=":", c="k", linewidth=2)
            ax_top.set_ylabel("z (m)")
            ax_top.set_title("Unshifted")
            ax_top.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
            ax_top.legend()

            # --- bottom: shifted ---
            ax_bot.plot(norm_r, z_m, label="Target Mirror")
            ax_bot.plot(norm_r, mr[0], label="Initial Mirror")
            ax_bot.plot(norm_r, final_shifted , c='k', linestyle="--", linewidth=5, label=label)
            ax_bot.set_xlabel(r"$r/w_0$")
            ax_bot.set_ylabel("z (m)")
            ax_bot.set_title("Shifted")
            ax_bot.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
            ax_bot.legend()

            if max(norm_r) > 2:
                ax_bot.set_xlim(xlim)
                ax_top.set_ylim(ylim)
                ax_bot.set_ylim(ylim)
            else:
                ax_bot.set_xlim(xmin=0)
                ax_top.set_ylim(ymin=0)
                ax_bot.set_ylim(ymin=0)

            plt.tight_layout()
            plt.savefig(path + "/mirror_comp_shifted.pdf")
            plt.close()

        else:
            final = mr[-1]
            plt.plot(norm_r, z_m, label="Target Mirror")
            plt.plot(norm_r, mr[0], label="Initial Mirror")
            plt.plot(norm_r, final - final[0], c='k', linestyle="--", linewidth=5, label=label)
            if max(norm_r) > 2:
                plt.xlim(xlim)
                plt.ylim(ylim)
            else:
                plt.xlim(xmin=0)
                plt.ylim(ymin=0)
            plt.xlabel(r"$r/w_0$")
            plt.ylabel("z (m)")
            plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
            plt.legend()
            plt.tight_layout()
            plt.savefig(path + "/mirror_comp.pdf")
            plt.close()


def plot_mirror_comp_each(path, r, w0, zm_t, mr, steps, xlim=(0, 2), ylim=(0, 2e-5), fontsize=30):
    """Save a mirror-vs-target comparison figure at regular step intervals.

    Iterates over the full mirror history and writes one PDF per sampled
    iteration to ``<path>/mirror/mirror_comp_<step>``.

    Parameters
    ----------
    path : str
        Output directory; images are written to the ``mirror/`` sub-directory.
    r : np.ndarray, shape (N,)
        Radial coordinate array (metres).
    w0 : float
        Beam waist radius used to normalise the radial axis (metres).
    zm_t : np.ndarray, shape (N,)
        Target mirror surface height profile (metres).
    mr : np.ndarray, shape (total_iterations, N)
        Mirror profile recorded at every optimisation iteration.
    steps : int
        Stride between saved frames; a figure is produced every ``steps``
        iterations.
    xlim : tuple of float, optional
        ``(xmin, xmax)`` in units of ``r/w0``. Applied only when
        ``max(r/w0) > 2``. Default is ``(0, 2)``.
    ylim : tuple of float, optional
        ``(ymin, ymax)`` in metres. Applied only when ``max(r/w0) > 2``.
        Default is ``(0, 2e-5)``.
    fontsize : int, optional
        Base font size for all text elements. Default is 30.

    Returns
    -------
    None
        Figures are saved to disk; progress is printed to stdout.
    """
    with plt.rc_context({"font.size": fontsize}):
        norm_r = r / w0

        for i, ref in enumerate(mr):
            if i % steps == 0:
                print("Mirror step: %d" % (i * steps))
                plt.figure(figsize=(11.52, 6.56), dpi=100)
                plt.plot(norm_r, zm_t, label="Target Mirror")
                plt.plot(norm_r, ref - ref[0], label="Mirror at step %d" % (i * steps))
                if max(norm_r) > 2:
                    plt.xlim(xlim)
                    plt.ylim(ylim)
                else:
                    plt.xlim(xmin=0)
                    plt.ylim(ymin=0)
                plt.legend()
                plt.xlabel(r"$r/w_0$")
                plt.ylabel("z (m)")
                plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
                plt.tight_layout()
                plt.savefig(path + "/mirror/mirror_comp_%d" % (i * steps), dpi=100)
                plt.close()


def plot_zernikes(path, zern, zern_grad=None, fontsize=30, step_label="Iteration"):
    """Two-panel figure: Zernike coefficient evolution (left) and a second panel (right).

    The right panel shows the per-step change (delta) in each coefficient,
    unless ``zern_grad`` is given, in which case it shows the actual
    optimizer gradient for each coefficient instead.

    Saves to ``<path>/zernikes.pdf``.

    Parameters
    ----------
    path : str
        Output directory where the PDF will be saved.
    zern : np.ndarray, shape (iterations, n_modes)
        Zernike coefficient history; rows are iterations, columns are modes.
    zern_grad : np.ndarray, shape (iterations, n_modes), optional
        Zernike coefficient gradient history. When given, the right panel
        plots this directly instead of ``np.diff(zern, axis=0)``. Default
        is ``None``.
    fontsize : int, optional
        Base font size for all text elements. Default is 30.
    step_label : str, optional
        Word used for the x-axis label ("Iteration" or "Generation").
        Default is "Iteration".

    Returns
    -------
    None
        Figure is saved to disk and the pyplot state is closed.
    """
    def _fmt_latex(val):
        if val == 0:
            return "$0$"
        exp = int(np.floor(np.log10(abs(val))))
        mantissa = abs(val) / 10 ** exp
        sign = "-" if val < 0 else ""
        return f"${sign}{mantissa:.2f} \\times 10^{{{exp}}}$"

    use_grad = zern_grad is not None
    delta_zern = zern_grad if use_grad else np.diff(zern, axis=0)

    with plt.rc_context({"font.size": fontsize}):
        n_modes = zern.shape[1]
        colors = plt.cm.nipy_spectral(np.linspace(0.1, 0.9, n_modes))
        iters = np.arange(len(zern))
        iters_d = iters if use_grad else iters[1:]

        final_abs_c = np.abs(zern[-1])
        min_final_c = np.min(final_abs_c[final_abs_c > 0]) if np.any(final_abs_c > 0) else 1e-6
        linthresh_c = min_final_c * 0.5

        abs_nonzero_d = np.abs(delta_zern[np.abs(delta_zern) > 0])
        linthresh_d = 10 ** np.floor(np.log10(np.percentile(abs_nonzero_d, 10))) if len(abs_nonzero_d) else 1e-3

        fig, (ax_c, ax_d) = plt.subplots(1, 2, figsize=(20, 9))

        for i in range(n_modes):
            ax_c.plot(iters, zern[:, i], color=colors[i], rasterized=True,
                      label=f"$a_{{{2*i}}}$ = {_fmt_latex(zern[-1, i])}")
            ax_d.plot(iters_d, delta_zern[:, i], color=colors[i], rasterized=True,
                      label=f"$a_{{{2*i}}}$ = {_fmt_latex(delta_zern[-1, i])}")
            if use_grad:
                ax_d.plot(iters_d[-1], delta_zern[-1, i], "o", color=colors[i], markersize=8, zorder=5)

        ax_c.axhline(0, color="k", linewidth=1, linestyle="--")
        ax_c.set_yscale("symlog", linthresh=linthresh_c)
        data_max_c = np.max(np.abs(zern))
        data_min_c = np.min(zern)
        ymax_c = 10 ** (int(np.floor(np.log10(data_max_c))) + 1)
        if data_min_c < 0:
            ymin_c = -10 ** (int(np.floor(np.log10(abs(data_min_c)))) + 1)
        else:
            ymin_c = -linthresh_c * 1.5
        ax_c.set_ylim(ymin_c, ymax_c)
        ax_c.set_xlabel(step_label)
        ax_c.set_ylabel(r"$a_i$")
        ax_c.set_title("Zernike Coefficient Evolution")
        ax_c.legend(loc="best")

        ax_d.axhline(0, color="k", linewidth=1, linestyle="--")
        ax_d.set_yscale("symlog", linthresh=linthresh_d)
        ax_d.set_xlabel(step_label)
        if use_grad:
            ax_d.set_ylabel(r"$\partial\mathcal{F}/\partial a_i$")
            ax_d.set_title("Zernike Gradient Evolution")
        else:
            ax_d.set_ylabel(r"$\Delta a_i$")
            ax_d.set_title(r"Zernike Coefficient Change ($\Delta a_i$)")
        ax_d.legend(loc="best")

        fig.tight_layout()
        fig.savefig(path + "/zernikes.pdf", bbox_inches="tight")
        plt.close(fig)


def plot_phi_f(path, phr_f, r, w0, ftf_t, fl, steps, fontsize=30):
    """Save individual focal-plane amplitude comparisons at regular step intervals.

    For every ``steps``-th iteration, writes a figure comparing the reflected
    amplitude ``phi_R(z=f)`` to the target amplitude ``phi_T(z=f)`` to
    ``<path>/pr_f/pr_f_<step>``.

    Parameters
    ----------
    path : str
        Output directory; images are written to the ``pr_f/`` sub-directory.
    phr_f : np.ndarray, shape (iterations, N)
        Reflected field amplitude at the focal plane for each iteration.
    r : np.ndarray, shape (N,)
        Radial coordinate array (metres).
    w0 : float
        Beam waist radius used to normalise the radial axis (metres).
    ftf_t : np.ndarray, shape (N, M)
        Target field amplitude on the full propagation grid; column ``fl``
        gives the focal-plane slice.
    fl : int
        Column index into ``ftf_t`` corresponding to the focal plane.
    steps : int
        Stride between saved frames.
    fontsize : int, optional
        Base font size for all text elements. Default is 30.

    Returns
    -------
    None
        Figures are saved to disk; progress is printed to stdout.
    """
    with plt.rc_context({"font.size": fontsize}):
        for i, ref in enumerate(phr_f):
            if i % steps == 0:
                print("PRPT_F step: %d" % (i * steps))
                plt.plot(r / w0, ftf_t[:, fl], label=r"$\phi_T(z=f)$")
                plt.plot(r / w0, ref, label=r"$\phi_R(z=f)$")
                plt.title("Step: %d" % (i * steps))
                plt.xlim((0, 0.2))
                plt.ylabel(r"$\phi$")
                plt.xlabel(r"$r/w_0$")
                plt.legend()
                plt.tight_layout()
                plt.savefig(path + "/pr_f/pr_f_%d" % (i * steps), dpi=100)
                plt.close()


def _select_by_convergence(data_iters, n_select, target, path=None):
    """Pick n_select frames at equal fractions of convergence toward mu=1.

    Loads mu = IrIt/Itsq from data.csv (column 1 / column 2) when path is
    given and row counts align; otherwise falls back to L2 shape-distance to
    target. Handles mu approaching 1 from either below or above by using
    |mu - 1| as the distance metric.
    """
    n = len(data_iters)
    if n <= n_select:
        return list(range(n))

    dist = None
    if path is not None:
        data_csv = os.path.join(path, "data.csv")
        if os.path.isfile(data_csv):
            raw = np.genfromtxt(data_csv, delimiter=",")
            if raw.ndim == 2 and raw.shape[1] >= 3 and raw.shape[0] == n:
                mu = raw[:, 1] / (raw[:, 2] + 1e-30)
                dist = np.abs(mu - 1.0)

    if dist is None:
        tgt = np.ravel(np.abs(target)).astype(float)
        tgt_norm = tgt / (np.linalg.norm(tgt) + 1e-30)
        dist = np.array([
            np.linalg.norm(np.ravel(np.abs(x)).astype(float) / (np.linalg.norm(np.ravel(np.abs(x))) + 1e-30) - tgt_norm)
            for x in data_iters
        ])

    progress = 1.0 - (dist - dist[-1]) / (dist[0] - dist[-1] + 1e-30)
    progress = np.clip(progress, 0, 1)
    targets = np.linspace(0, 1, n_select)[:-1]
    chosen = sorted({int(np.argmin(np.abs(progress - t))) for t in targets} | {n - 1})
    return chosen


def plot_phi_f_over_iterations(path, phr_f, r, w0, ftf_t, fl, skip, steps=None, fontsize=30, step_label="Iteration"):
    """Overlay focal-plane amplitude at selected iterations on a single plot.

    If ``steps`` is ``None``, iterations are chosen on a geometric sequence so
    that early and late behaviour are both represented. Saves to
    ``<path>/phi_f_over_iterations.pdf``.

    Parameters
    ----------
    path : str
        Output directory where the PDF will be saved.
    phr_f : np.ndarray, shape (iterations, N)
        Reflected field amplitude at the focal plane for each iteration.
    r : np.ndarray, shape (N,)
        Radial coordinate array (metres).
    w0 : float
        Beam waist radius used to normalise the radial axis (metres).
    ftf_t : np.ndarray, shape (N, M)
        Target field amplitude on the full propagation grid; column ``fl``
        gives the focal-plane slice.
    fl : int
        Column index into ``ftf_t`` corresponding to the focal plane.
    steps : list of int or None, optional
        Specific iteration indices to include. If ``None``, up to 4 indices
        are selected via ``np.linspace``. Default is ``None``.
    fontsize : int, optional
        Base font size for all text elements. Default is 30.

    Returns
    -------
    None
        Figure is saved to disk and the pyplot state is closed.
    """
    with plt.rc_context({"font.size": fontsize}):
        n = len(phr_f)
        if steps is None:
            steps = _select_by_convergence(phr_f, min(4, n), ftf_t[:, fl], path=path)
        else:
            steps = [s for s in steps if s < n]

        line_colors = plt.cm.nipy_spectral(np.linspace(0.1, 0.9, len(steps)))
        for i, num in enumerate(steps):
            plt.plot(
                r / w0,
                phr_f[num],
                c=line_colors[i],
                alpha=0.85,
                label=r"$\phi_R$ step %d" % (num * skip),
            )
        plt.plot(r / w0, ftf_t[:, fl], c="k", linestyle="--", linewidth=3, label=r"$\phi_T(z=f)$")
        plt.xlim((0, 0.2))
        plt.ylabel(r"$\phi$")
        plt.xlabel(r"$r/w_0$")
        plt.legend()
        plt.title(f"Focal-plane Amplitude over {step_label}s")
        plt.tight_layout()
        plt.savefig(path + "/phi_f_over_iterations.pdf")
        plt.close()


def plot_I_f(path, ir_f, r, w0, it_f, steps, fontsize=30):
    """Save individual focal-plane intensity comparisons at regular step intervals.

    For every ``steps``-th iteration, writes a figure comparing the reflected
    intensity ``I_R(z=f)`` to the target intensity ``I_T(z=f)`` to
    ``<path>/I_f/I_f_<step>``.

    Parameters
    ----------
    path : str
        Output directory; images are written to the ``I_f/`` sub-directory.
    ir_f : np.ndarray, shape (iterations, N)
        Reflected field intensity at the focal plane for each iteration.
    r : np.ndarray, shape (N,)
        Radial coordinate array (metres).
    w0 : float
        Beam waist radius used to normalise the radial axis (metres).
    it_f : np.ndarray, shape (N,)
        Target intensity profile at the focal plane.
    steps : int
        Stride between saved frames.
    fontsize : int, optional
        Base font size for all text elements. Default is 30.

    Returns
    -------
    None
        Figures are saved to disk; progress is printed to stdout.
    """
    with plt.rc_context({"font.size": fontsize}):
        for i, ref in enumerate(ir_f):
            if i % steps == 0:
                print("I_F step: %d" % (i * steps))
                plt.plot(r / w0, it_f, label=r"$I_T(z=f)$")
                plt.plot(r / w0, ref, label=r"$I_R(z=f)$")
                plt.title("Step: %d" % (i * steps))
                plt.xlim((0, 0.2))
                plt.ylabel(r"$\phi$")
                plt.xlabel(r"$r/w_0$")
                plt.legend()
                plt.tight_layout()
                plt.savefig(path + "/I_f/I_f_%d" % (i * steps), dpi=100)
                plt.close()


def plot_I_f_over_iterations(path, ir_f, r, w0, it_f, skip, steps=None, fontsize=30, step_label="Iteration"):
    """Overlay focal-plane intensity at selected iterations on a single plot.

    Saves to ``<path>/I_f_over_iterations.pdf``.

    Parameters
    ----------
    path : str
        Output directory where the PDF will be saved.
    ir_f : np.ndarray, shape (iterations, N)
        Reflected field intensity at the focal plane for each iteration.
    r : np.ndarray, shape (N,)
        Radial coordinate array (metres).
    w0 : float
        Beam waist radius used to normalise the radial axis (metres).
    it_f : np.ndarray, shape (N,)
        Target intensity profile at the focal plane. Must already be scaled
        consistently with ``ir_f`` (i.e. normalised by the same global
        target-intensity peak used when ``ir_f`` was saved during the run) --
        this function plots it as-is rather than re-normalising it locally.
    steps : list of int or None, optional
        Specific iteration indices to include. If ``None``, up to 4 indices
        are selected via ``np.linspace``. Default is ``None``.
    fontsize : int, optional
        Base font size for all text elements. Default is 30.

    Returns
    -------
    None
        Figure is saved to disk and the pyplot state is closed.
    """
    with plt.rc_context({"font.size": fontsize}):
        n = len(ir_f)
        if steps is None:
            steps = _select_by_convergence(ir_f, min(4, n), it_f, path=path)
        else:
            steps = [s for s in steps if s < n]

        target_f = it_f
        line_colors = plt.cm.nipy_spectral(np.linspace(0.1, 0.9, len(steps)))
        for i, num in enumerate(steps):
            plt.plot(
                r / w0,
                ir_f[num],
                c=line_colors[i],
                alpha=0.85,
                label=r"$I_R$ step %d" % (num * skip),
            )
        plt.plot(r / w0, target_f, c="k", linestyle="--", linewidth=3, label=r"$I_T(z=f)$")
        plt.xlim((0, 0.2))
        plt.ylabel("Focal-plane Intensity")
        plt.xlabel(r"$r/w_0$")
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        plt.legend()
        plt.title(f"Focal-plane Intensity over {step_label}s")
        plt.tight_layout()
        plt.savefig(path + "/I_f_over_iterations.pdf")
        plt.close()


# In[6]:


def plot_I_OA(path, ir_oa, z, f, it, label_skip, fontsize=30):
    """Save one on-axis intensity comparison frame per saved iteration.

    Writes, for every row of ``ir_oa``, a figure comparing the reflected
    on-axis intensity ``I_R`` to the target ``I_T`` as a function of
    propagation distance ``z - f``, saved to ``<path>/I_OA/I_oa_<step>``.

    Parameters
    ----------
    path : str
        Output directory; images are written to the ``I_OA/`` sub-directory.
    ir_oa : np.ndarray, shape (iterations, M)
        Reflected on-axis intensity for each saved optimisation iteration.
    z : np.ndarray, shape (M,)
        Propagation axis coordinate array (metres).
    f : float
        Focal length; used to shift the axis so the focus is at the origin
        (metres).
    it : np.ndarray, shape (M,)
        Target on-axis intensity profile.
    label_skip : int
        Real-step multiplier: row ``i`` of ``ir_oa`` corresponds to real
        optimisation step ``i * label_skip`` (1 if every step was saved to
        ``ir_oa``, or the run's save interval if rows were themselves
        subsampled at write time). Used only for the step number shown in
        the title/filename -- every row still gets its own frame.
    fontsize : int, optional
        Base font size for all text elements. Default is 30.

    Returns
    -------
    None
        Figures are saved to disk; progress is printed to stdout.
    """
    with plt.rc_context({"font.size": fontsize}):
        for i, ref in enumerate(ir_oa):
            step = i * label_skip
            print("Int_OA step: %d" % step)
            plt.plot((z - f) * 100, it, label=r"$I_T$")
            plt.plot((z - f) * 100, ref, label=r"$I_R$")
            plt.title("Step: %d" % step)
            plt.ylabel("On-axis Intensity (arb. units)")
            plt.xlabel(r"$z-f$ (cm)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(path + "/I_OA/I_oa_%d" % step, dpi=100)
            plt.close()


def plot_I_OA_over_iterations(path, ir_oa, z, f, it, skip, steps=None, xlim=None, fontsize=30, step_label="Iteration"):
    """Overlay on-axis intensity at selected iterations on a single plot.

    When the output path contains the string ``"int"``, the target is
    peak-normalised before plotting. Saves to
    ``<path>/I_OA_over_iterations.pdf``.

    Parameters
    ----------
    path : str
        Output directory where the PDF will be saved. If ``"int"`` appears in
        the path, ``it`` is peak-normalised.
    ir_oa : np.ndarray, shape (iterations, M)
        Reflected on-axis intensity for each optimisation iteration.
    z : np.ndarray, shape (M,)
        Propagation axis coordinate array (metres).
    f : float
        Focal length used to shift the axis so the focus is at the origin
        (metres).
    it : np.ndarray, shape (M,)
        Target on-axis intensity profile.
    steps : list of int or None, optional
        Specific iteration indices to include. If ``None``, up to 4 indices
        are selected via ``np.linspace``. Default is ``None``.
    fontsize : int, optional
        Base font size for all text elements. Default is 30.

    Returns
    -------
    None
        Figure is saved to disk and the pyplot state is closed.
    """
    with plt.rc_context({"font.size": fontsize}):
        n = len(ir_oa)
        if steps is None:
            steps = _select_by_convergence(ir_oa, min(4, n), it, path=path)
        else:
            steps = [s for s in steps if s < n]

        target_oa = it / np.max(it) if "int" in path else it
        line_colors = plt.cm.nipy_spectral(np.linspace(0.1, 0.9, len(steps)))
        for i, num in enumerate(steps):
            plt.plot(
                (z - f) * 100,
                ir_oa[num],
                c=line_colors[i],
                alpha=0.85,
                label=r"$I_R$ step %d" % (num * skip),
            )
        plt.plot((z - f) * 100, target_oa, c="k", linestyle="--", linewidth=3, label=r"$I_T$")
        plt.ylabel("On-axis Intensity (arb. units)")
        plt.xlabel(r"$z-f$ (cm)")
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        if xlim is not None:
            plt.xlim(xlim)
        plt.legend()
        plt.title(f"Reflected Intensity over {step_label}s")
        plt.tight_layout()
        plt.savefig(path + "/I_OA_over_iterations.pdf")
        plt.close()


def plot_I_OA_over_iterations_zoom(path, ir_oa, z, f, it, skip, xlim, steps=None, fontsize=30, step_label="Iteration"):
    """Like plot_I_OA_over_iterations but with a fixed window and y-axis clipped to [0, 1.5].

    Saves to ``<path>/I_OA_over_iterations_zoom.pdf``.
    """
    with plt.rc_context({"font.size": fontsize}):
        n = len(ir_oa)
        if steps is None:
            steps = _select_by_convergence(ir_oa, min(4, n), it, path=path)
        else:
            steps = [s for s in steps if s < n]

        it_max = np.max(it)
        target_oa = it / it_max
        ir_scale = 1.0 if "int" in path else it_max
        line_colors = plt.cm.nipy_spectral(np.linspace(0.1, 0.9, len(steps)))
        for i, num in enumerate(steps):
            plt.plot(
                (z - f) * 100,
                ir_oa[num] / ir_scale,
                c=line_colors[i],
                alpha=0.85,
                label=r"$I_R$ step %d" % (num * skip),
            )
        plt.plot((z - f) * 100, target_oa, c="k", linestyle="--", linewidth=3, label=r"$I_T$")
        plt.ylabel("Normalized On-axis Intensity")
        plt.xlabel(r"$z-f$ (cm)")
        plt.xlim(xlim)
        plt.ylim(0, 1.5)
        plt.legend()
        plt.title(f"Reflected Intensity over {step_label}s")
        plt.tight_layout()
        plt.savefig(path + "/I_OA_over_iterations_zoom.pdf")
        plt.close()


def plot_phi_OA_over_iterations(path, phr_oa, z, f, ftf_t, skip, steps=None, xlim=None, fontsize=30, step_label="Iteration"):
    """Overlay on-axis amplitude at selected iterations on a single plot.

    Saves to ``<path>/phi_OA_over_iterations.pdf``.

    Parameters
    ----------
    path : str
        Output directory where the PDF will be saved.
    phr_oa : np.ndarray, shape (iterations, M)
        Reflected on-axis field amplitude for each optimisation iteration.
    z : np.ndarray, shape (M,)
        Propagation axis coordinate array (metres).
    f : float
        Focal length used to shift the axis so the focus is at the origin
        (metres).
    ftf_t : np.ndarray, shape (N, M)
        Target field amplitude on the full propagation grid; row 0 gives the
        on-axis (r=0) slice.
    steps : list of int or None, optional
        Specific iteration indices to include. If ``None``, up to 4 indices
        are selected via ``np.linspace``. Default is ``None``.
    fontsize : int, optional
        Base font size for all text elements. Default is 30.

    Returns
    -------
    None
        Figure is saved to disk and the pyplot state is closed.
    """
    with plt.rc_context({"font.size": fontsize}):
        n = len(phr_oa)
        if steps is None:
            steps = _select_by_convergence(phr_oa, min(4, n), ftf_t[0, :], path=path)
        else:
            steps = [s for s in steps if s < n]

        line_colors = plt.cm.nipy_spectral(np.linspace(0.1, 0.9, len(steps)))
        for i, num in enumerate(steps):
            plt.plot(
                (z - f) * 100,
                phr_oa[num],
                c=line_colors[i],
                alpha=0.85,
                label=r"$\phi_R$ step %d" % (num * skip),
            )
        plt.plot((z - f) * 100, ftf_t[0, :], c="k", linestyle="--", linewidth=3, label=r"$\phi_T$")
        plt.ylabel("On-axis Amplitude")
        plt.xlabel(r"$z-f$ (cm)")
        if xlim is not None:
            plt.xlim(xlim)
        plt.legend()
        plt.title(f"On-axis Amplitude over {step_label}s")
        plt.tight_layout()
        plt.savefig(path + "/phi_OA_over_iterations.pdf")
        plt.close()


def plot_phi_OA(path, phr_oa, z, f, ftf_t, steps, fontsize=30):
    """Save individual on-axis amplitude comparisons at regular step intervals.

    For every ``steps``-th iteration, writes a figure comparing the reflected
    on-axis amplitude ``phi_R`` to the target ``phi_T`` to
    ``<path>/pr_oa/pr_oa_<step>``.

    Parameters
    ----------
    path : str
        Output directory; images are written to the ``pr_oa/`` sub-directory.
    phr_oa : np.ndarray, shape (iterations, M)
        Reflected on-axis field amplitude for each optimisation iteration.
    z : np.ndarray, shape (M,)
        Propagation axis coordinate array (metres).
    f : float
        Focal length used to shift the axis so the focus is at the origin
        (metres).
    ftf_t : np.ndarray, shape (N, M)
        Target field amplitude on the full propagation grid; row 0 gives the
        on-axis (r=0) slice.
    steps : int
        Stride between saved frames.
    fontsize : int, optional
        Base font size for all text elements. Default is 30.

    Returns
    -------
    None
        Figures are saved to disk; progress is printed to stdout.
    """
    with plt.rc_context({"font.size": fontsize}):
        for i, ref in enumerate(phr_oa):
            if i % steps == 0:
                print("PRPT_OA step: %d" % (i * steps))
                plt.plot((z - f) * 100, ftf_t[0, :], label=r"$\phi_T$")
                plt.plot((z - f) * 100, ref, label=r"$\phi_R$")
                plt.title("Step: %d" % (i * steps))
                plt.ylabel("On-axis Amplitude")
                plt.xlabel(r"$z-f$ (cm)")
                plt.legend()
                plt.tight_layout()
                plt.savefig(path + "/pr_oa/pr_oa_%d" % (i * steps), dpi=100)
                plt.close()


# # Show each step size

# In[37]:


def plot_da_vs_grad(path, gr, r, w0, zern, n, skip, fontsize=30):
    """Compare the raw mirror gradient to its Zernike projection at regular intervals.

    For every ``skip``-th iteration, overlays the adjoint gradient ``df/dz``
    with the Zernike-fitted gradient to show how well the polynomial basis
    captures the computed update. Saves frames to ``<path>/g/g_<step>``.

    Parameters
    ----------
    path : str
        Output directory; images are written to the ``g/`` sub-directory.
    gr : np.ndarray, shape (iterations, N)
        Raw adjoint gradient (mirror deformation per step) at each iteration.
    r : np.ndarray, shape (N,)
        Radial coordinate array (metres).
    w0 : float
        Beam waist radius used to normalise the radial axis (metres).
    zern : np.ndarray, shape (iterations, n_modes)
        Zernike coefficient updates recorded at each iteration.
    n : int
        Number of Zernike modes passed to ``calc_Z`` for constructing the
        design matrix.
    skip : int
        Stride between saved frames.
    fontsize : int, optional
        Base font size for all text elements. Default is 30.

    Returns
    -------
    None
        Figures are saved to disk; progress is printed to stdout.
    """
    with plt.rc_context({"font.size": fontsize}):
        i = 0
        Z_fit = calc_Z(r / np.max(r), n)
        dfda = Z_fit @ zern.T
        norm_r = r / w0
        for dzm, da in zip(gr, dfda.T):
            if i % skip == 0:
                print("G step: %d" % (i * skip))
                plt.scatter(r / w0, dzm, label="df/dz")
                plt.scatter(r / w0, da, label="Zernike Fit on df/dz")
                plt.title("Step: %d" % (i * steps))
                plt.ylabel(r"$\delta z/\lambda$")
                plt.xlabel(r"$r/w_0$")
                plt.axhline(0, linestyle="--", c="k")
                if max(norm_r) > 2:
                    plt.xlim((0, 2))
                else:
                    plt.xlim(xmin=0)
                plt.legend()
                plt.tight_layout()
                plt.savefig(path + "/g/g_%d" % (i * skip), dpi=100)
                plt.close()
            i += 1


def plot_psi(path, psi, r, w0, steps, fontsize=30):
    """Save the real part of the adjoint field psi(r) at regular step intervals.

    Writes one figure per sampled iteration to ``<path>/psi/psi_<step>``.

    Parameters
    ----------
    path : str
        Output directory; images are written to the ``psi/`` sub-directory.
    psi : np.ndarray, shape (iterations, N), complex
        Adjoint field at the mirror plane recorded at each iteration.
    r : np.ndarray, shape (N,)
        Radial coordinate array (metres).
    w0 : float
        Beam waist radius used to normalise the radial axis (metres).
    steps : int
        Stride between saved frames.
    fontsize : int, optional
        Base font size for all text elements. Default is 30.

    Returns
    -------
    None
        Figures are saved to disk; progress is printed to stdout.
    """
    with plt.rc_context({"font.size": fontsize}):
        for i, ref in enumerate(psi):
            if i % steps == 0:
                print("Psi step: %d" % (i * steps))
                plt.plot(r / w0, np.real(ref))
                plt.title("Step: %d" % (i * steps))
                plt.ylabel(r"$\psi$")
                plt.xlabel(r"$r/w_0$")
                plt.tight_layout()
                plt.savefig(path + "/psi/psi_%d" % (i * steps), dpi=100)
                plt.close()


def plot_source(path, S, z, f, steps, fontsize=30):
    """Save the real part of the adjoint source S(z) at regular step intervals.

    Writes one figure per sampled iteration to ``<path>/S/S_<step>``.

    Parameters
    ----------
    path : str
        Output directory; images are written to the ``S/`` sub-directory.
    S : np.ndarray, shape (iterations, M), complex
        Adjoint source field along the propagation axis recorded at each
        iteration.
    z : np.ndarray, shape (M,)
        Propagation axis coordinate array (metres).
    f : float
        Focal length used to normalise the z-axis (metres).
    steps : int
        Stride between saved frames.
    fontsize : int, optional
        Base font size for all text elements. Default is 30.

    Returns
    -------
    None
        Figures are saved to disk; progress is printed to stdout.
    """
    with plt.rc_context({"font.size": fontsize}):
        for i, ref in enumerate(S):
            if i % steps == 0:
                print("S step: %d" % (i * steps))
                plt.plot(z / f, np.real(ref))
                plt.title("Step: %d" % (i * steps))
                plt.ylabel(r"$S$")
                plt.xlabel(r"$z/f$")
                plt.tight_layout()
                plt.savefig(path + "/S/S_%d" % (i * steps), dpi=100)
                plt.close()


def plot_step0_intensity(path, ir_oa, ir_f, z, f, r, w0, It, fl, fontsize=30):
    """Two-panel figure (stacked vertically) of I_OA and I_f at step 0.

    Saves to ``<path>/step0_intensity.pdf``.
    """
    with plt.rc_context({"font.size": fontsize}):
        fig, (ax_oa, ax_f) = plt.subplots(2, 1, figsize=(16, 18))

        # It arrives already correctly scaled (normalized by the same global
        # peak used to save ir_oa/ir_f when "int" in path, raw otherwise) --
        # re-normalizing it_f_plot by its own local max here would use a
        # different constant than ir_f was actually saved with (the global
        # peak isn't generally at this one fixed z-slice), same bug as
        # plot_I_f_over_iterations used to have.
        it_oa = It[0, :]
        it_f_plot = It[:, fl]

        ax_oa.plot((z - f) * 100, it_oa, label=r"$I_T$")
        ax_oa.plot((z - f) * 100, ir_oa[0], label=r"$I_R$ step 0")
        ax_oa.set_ylabel("On-axis Intensity (arb. units)")
        ax_oa.set_xlabel(r"$z-f$ (cm)")
        ax_oa.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax_oa.set_title(r"$I_{OA}$ at Step 0")
        ax_oa.legend()

        fl_label = r"$I_T(z=f+0.5L)$" if "axi" in path else r"$I_T(z=f)$"
        ax_f.plot(r / w0, it_f_plot, label=fl_label)
        ax_f.plot(r / w0, ir_f[0], label=r"$I_R$ step 0")
        ax_f.set_xlim((0, 0.2))
        ax_f.set_ylabel("Focal-plane Intensity")
        ax_f.set_xlabel(r"$r/w_0$")
        ax_f.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax_f.set_title(r"$I_f$ at Step 0")
        ax_f.legend()

        fig.tight_layout()
        fig.savefig(path + "/step0_intensity.pdf")
        plt.close(fig)


def plot_step0_phi(path, phr_oa, phr_f, z, f, r, w0, ftf_t, fl, fontsize=30):
    """Two-panel figure (stacked vertically) of phi_OA and phi_f at step 0.

    Saves to ``<path>/step0_phi.pdf``.
    """
    with plt.rc_context({"font.size": fontsize}):
        fig, (ax_oa, ax_f) = plt.subplots(2, 1, figsize=(16, 18))

        ax_oa.plot((z - f) * 100, ftf_t[0, :], label=r"$\phi_T$")
        ax_oa.plot((z - f) * 100, phr_oa[0], label=r"$\phi_R$ step 0")
        ax_oa.set_ylabel("On-axis Amplitude")
        ax_oa.set_xlabel(r"$z-f$ (cm)")
        ax_oa.set_title(r"$\phi_{OA}$ at Step 0")
        ax_oa.legend()

        fl_label_phi = r"$\phi_T(z=f+0.5L)$" if "axi" in path else r"$\phi_T(z=f)$"
        ax_f.plot(r / w0, ftf_t[:, fl], label=fl_label_phi)
        ax_f.plot(r / w0, phr_f[0], label=r"$\phi_R$ step 0")
        ax_f.set_xlim((0, 0.2))
        ax_f.set_ylabel(r"$\phi$")
        ax_f.set_xlabel(r"$r/w_0$")
        ax_f.set_title(r"$\phi_f$ at Step 0")
        ax_f.legend()

        fig.tight_layout()
        fig.savefig(path + "/step0_phi.pdf")
        plt.close(fig)


def plot_psi_and_source(path, psi, r, w0, S, z, f, skip, fontsize=30):
    """Save side-by-side panels of the adjoint source S(z) and mirror field psi(r).

    For every ``skip``-th iteration, produces a two-panel figure showing
    ``S(z)`` on the left and ``psi(r)`` on the right. Saves frames to
    ``<path>/psi_S/psi_S_<step>``.

    Parameters
    ----------
    path : str
        Output directory; images are written to the ``psi_S/`` sub-directory.
    psi : np.ndarray, shape (iterations, N), complex
        Adjoint field at the mirror plane recorded at each iteration.
    r : np.ndarray, shape (N,)
        Radial coordinate array (metres).
    w0 : float
        Beam waist radius used to normalise the radial axis for the psi panel
        (metres).
    S : np.ndarray, shape (iterations, M), complex
        Adjoint source field along the propagation axis recorded at each
        iteration.
    z : np.ndarray, shape (M,)
        Propagation axis coordinate array (metres).
    f : float
        Focal length used to normalise the z-axis for the S panel (metres).
    skip : int
        Stride between saved frames.
    fontsize : int, optional
        Base font size for all text elements. Default is 30.

    Returns
    -------
    None
        Figures are saved to disk; progress is printed to stdout.
    """
    with plt.rc_context({"font.size": fontsize}):
        for i, (psi_ref, S_ref) in enumerate(zip(psi, S)):
            if i % skip == 0:
                print(f"Plotting step: {i * skip}")

                fig, axes = plt.subplots(
                    1, 2, figsize=(16, 9), dpi=100, gridspec_kw={"wspace": 0.25}
                )

                axes[0].plot(z / f, np.real(S_ref), color="green")
                axes[0].set_title(r"$S(z)$")
                axes[0].set_xlabel(r"$z/f$")
                axes[0].set_ylabel(r"$S$")

                axes[1].plot(r / w0, np.real(psi_ref), color="blue")
                axes[1].set_title(r"$\psi(r)$")
                axes[1].set_xlabel(r"$r/w_0$")
                axes[1].set_ylabel(r"$\psi$")

                fig.suptitle(f"Step: {i * skip}")
                plt.subplots_adjust(left=0.1, right=0.97, top=0.88, bottom=0.12)
                plt.savefig(path + "/psi_S/psi_S_%d" % (i * skip), dpi=100)
                plt.close()


def plot_gradient(path, gradient, r, w0, crop_rmax=2, linthresh=10, linscale=1.0, fontsize=26, step_label="Iteration"):
    """Plot the gradient history as a 2D colormap over radial position and iterations.

    Always saves a full-range linear plot and a SymLogNorm version. When
    ``max(r/w0) > crop_rmax``, also saves cropped versions zoomed to
    ``r/w0 in [0, crop_rmax]``. Saves to ``gradient_full.pdf``,
    ``gradient_full_symlognorm.pdf``, and (when cropped) ``gradient_zoom.pdf``,
    ``gradient_zoom_symlognorm.pdf``.

    Parameters
    ----------
    path : str
        Output directory where PDFs will be saved.
    gradient : np.ndarray, shape (n_iters, n_r)
        Gradient history; rows are saved iterations, columns are radial points.
    r : np.ndarray, shape (n_r,)
        Radial coordinate array (metres).
    w0 : float
        Beam waist radius (metres) used to normalise the radial axis.
    crop_rmax : float, optional
        Upper r/w0 limit for the zoomed plot. Only used when ``max(r/w0) > crop_rmax``.
        Default is 2.
    linthresh : float, optional
        Linear threshold for SymLogNorm. Default is 10.
    linscale : float, optional
        Linear scale for SymLogNorm. Default is 1.0.
    fontsize : int, optional
        Base font size for all text elements. Default is 26.

    Returns
    -------
    None
        Figures are saved to disk and the pyplot state is closed.
    """
    with plt.rc_context({"font.size": fontsize}):
        rmax = np.max(r / w0)
        n_iters, n_r = gradient.shape

        def _draw(data, n_cols, r_hi, x_fmt, use_symlognorm, title):
            abs_data = np.abs(data)
            vabs = np.percentile(abs_data, 99) if not use_symlognorm else np.max(abs_data)

            fig, ax_map = plt.subplots(figsize=(16, 9))

            if use_symlognorm:
                norm = SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=-vabs, vmax=vabs)
                im = ax_map.imshow(data, origin="lower", aspect="auto", cmap="RdBu_r", norm=norm)
            else:
                im = ax_map.imshow(data, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vabs, vmax=vabs)

            x_ticks = np.linspace(0, n_cols, 6)
            x_labels = np.linspace(0, r_hi, 6)
            y_ticks = np.linspace(0, n_iters, 11)
            y_labels = np.linspace(0, n_iters, 11)
            ax_map.set_xticks(x_ticks)
            ax_map.set_xticklabels([x_fmt.format(x) for x in x_labels])
            ax_map.set_yticks(y_ticks)
            ax_map.set_yticklabels([f"{y:.0f}" for y in y_labels])
            ax_map.set_xlabel(r"$r/w_0$")
            ax_map.set_ylabel(step_label + "s")
            ax_map.set_title(title)

            cb = fig.colorbar(im, ax=ax_map, shrink=0.9)
            cb.set_label(r"$\partial \mathcal{F}/\partial z$")
            if use_symlognorm:
                cb.ax.axhline(linthresh, color="k", linewidth=0.8, linestyle="--")
                cb.ax.axhline(-linthresh, color="k", linewidth=0.8, linestyle="--")
                cb.ax.text(1.05, linthresh, f"{linthresh:.3g}", transform=cb.ax.get_yaxis_transform(),
                           va="center", fontsize=fontsize * 0.55)
                cb.ax.text(1.05, -linthresh, f"{-linthresh:.3g}", transform=cb.ax.get_yaxis_transform(),
                           va="center", fontsize=fontsize * 0.55)

            fig.tight_layout()
            return fig

        fig = _draw(gradient, n_r, rmax, "{:.1f}", use_symlognorm=False, title="Gradient History (Linear)")
        fig.savefig(path + "/gradient_full.pdf")
        plt.close(fig)

        fig = _draw(gradient, n_r, rmax, "{:.1f}", use_symlognorm=True, title="Gradient History (SymLog)")
        fig.savefig(path + "/gradient_full_symlognorm.pdf")
        plt.close(fig)

        if rmax > crop_rmax:
            crop_idx = int(round(crop_rmax / rmax * n_r))
            cropped = gradient[:, :crop_idx]

            fig = _draw(cropped, crop_idx, crop_rmax, "{:.1f}", use_symlognorm=False,
                        title=f"Gradient History (Linear, $r/w_0 \\leq {crop_rmax}$)")
            fig.savefig(path + "/gradient_zoom.pdf")
            plt.close(fig)

            fig = _draw(cropped, crop_idx, crop_rmax, "{:.1f}", use_symlognorm=True,
                        title=f"Gradient History (SymLog, $r/w_0 \\leq {crop_rmax}$)")
            fig.savefig(path + "/gradient_zoom_symlognorm.pdf")
            plt.close(fig)


def plot_zernike_trajectory(path, zern, data, coeff_i=1, coeff_j=2, subsample=10,
                            arrow_every=20, fontsize=30):
    """Plot the optimizer trajectory through the 2D subspace of two Zernike coefficients.

    Scatter points are colored by mu (overlap integral). Direction arrows show
    which way the optimizer is moving. Start and end are marked distinctly.
    Saves to ``<path>/zernike_trajectory.pdf``.

    Parameters
    ----------
    path : str
        Output directory.
    zern : np.ndarray, shape (n_steps, n_coeffs)
        Zernike coefficient history (one row per iteration).
    data : np.ndarray, shape (n_steps, 3)
        Columns are Irsq, IrIt, Itsq. Used to compute mu at each step.
    coeff_i : int
        0-indexed column of the x-axis coefficient. Default 1 (2nd coefficient).
    coeff_j : int
        0-indexed column of the y-axis coefficient. Default 2 (3rd coefficient).
    subsample : int
        Plot every Nth step to avoid overplotting. Default 10.
    arrow_every : int
        Draw a direction arrow every Nth subsampled point. Default 20.
    fontsize : int
        Base font size. Default 30.
    """
    with plt.rc_context({"font.size": fontsize}):
        rr, rt, tt = data[:, 0], data[:, 1], data[:, 2]
        mu = rt / tt

        xi = zern[::subsample, coeff_i]
        xj = zern[::subsample, coeff_j]
        mu_s = mu[::subsample]

        cmap = plt.cm.plasma
        norm = mpl.colors.Normalize(vmin=0, vmax=1)

        fig, ax = plt.subplots(figsize=(12, 10))

        # thin path line
        ax.plot(xi, xj, color="gray", lw=0.8, alpha=0.4, zorder=1)

        # scatter colored by mu
        sc = ax.scatter(xi, xj, c=mu_s, cmap=cmap, norm=norm, s=18, zorder=2,
                        linewidths=0)

        # direction arrows
        for k in range(0, len(xi) - 1, arrow_every):
            dx = xi[k + 1] - xi[k]
            dy = xj[k + 1] - xj[k]
            ax.annotate("", xy=(xi[k] + dx, xj[k] + dy), xytext=(xi[k], xj[k]),
                        arrowprops=dict(arrowstyle="-|>", color=cmap(norm(mu_s[k])),
                                        lw=1.5, mutation_scale=14),
                        zorder=3)

        # start and end markers
        ax.scatter(xi[0], xj[0], marker="*", s=400, color="lime", edgecolors="k",
                   linewidths=1, zorder=5, label="Start")
        ax.scatter(xi[-1], xj[-1], marker="D", s=150, color="red", edgecolors="k",
                   linewidths=1, zorder=5, label="End")

        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label(r"$\mu$")
        cb.ax.axhline(1.0, color="white", lw=1.2, ls="--")

        ax.set_xlabel(rf"$a_{{{coeff_i + 1}}}$")
        ax.set_ylabel(rf"$a_{{{coeff_j + 1}}}$")
        ax.set_title(r"Optimizer trajectory in Zernike coefficient space")
        ax.legend(fontsize=fontsize * 0.7, markerscale=0.8)
        ax.grid(True, alpha=0.25)

        fig.tight_layout()
        fig.savefig(path + "/zernike_trajectory.pdf")
        plt.close(fig)
