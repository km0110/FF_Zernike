#!/usr/bin/env python
# coding: utf-8

# In[1]:

import matplotlib as mpl
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
def plot_fom_mu(path, fom, mu, fontsize=30):
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
        ax1.set_xlabel("Iterations")
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


def plot_percent_change_mu(path, mu, fontsize=30):
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
        plt.xlabel("Iteration")
        plt.ylabel(r"$|\%\ \mathrm{Change\ in}\ \mu|$")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(path + "/pc_mu.pdf")
        plt.close()


def plot_integrals(path, prsq, prpt, ptsq, fontsize=30):
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
        plt.xlabel("Iterations")
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
                print("Mirror step: %d" % i)
                plt.figure(figsize=(11.52, 6.56), dpi=100)
                plt.plot(norm_r, zm_t, label="Target Mirror")
                plt.plot(norm_r, ref - ref[0], label="Mirror at step %d" % i)
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
                plt.savefig(path + "/mirror/mirror_comp_%d" % i, dpi=100)
                plt.close()


def plot_zernikes(path, zern, fontsize=30):
    """Plot the evolution of even-order radial Zernike coefficients over iterations.

    Each column of ``zern`` corresponds to one Zernike mode and is drawn as a
    separate line. Saves to ``<path>/zernikes.pdf``.

    Parameters
    ----------
    path : str
        Output directory where the PDF will be saved.
    zern : np.ndarray, shape (iterations, n_modes)
        Zernike coefficient history; rows are iterations, columns are modes
        ordered as Z(0,0), Z(2,0), Z(4,0), …
    fontsize : int, optional
        Base font size for all text elements. Default is 30.

    Returns
    -------
    None
        Figure is saved to disk and the pyplot state is closed.
    """
    with plt.rc_context({"font.size": fontsize}):
        for i in range(zern.shape[1]):
            plt.plot(zern[:, i], label=f"Zernike Z{2*i},0")
        plt.xlabel("Iteration")
        plt.ylabel("Coefficient")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path + "/zernikes.pdf")
        plt.close()


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
                print("PRPT_F step: %d" % i)
                plt.plot(r / w0, ftf_t[:, fl], label=r"$\phi_T(z=f)$")
                plt.plot(r / w0, ref, label=r"$\phi_R(z=f)$")
                plt.title("Step: %d" % i)
                plt.xlim((0, 0.2))
                plt.ylabel(r"$\phi$")
                plt.xlabel(r"$r/w_0$")
                plt.legend()
                plt.tight_layout()
                plt.savefig(path + "/pr_f/pr_f_%d" % i, dpi=100)
                plt.close()


def plot_phi_f_over_iterations(path, phr_f, r, w0, ftf_t, fl, steps=None, fontsize=30):
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
        are selected via ``np.geomspace``. Default is ``None``.
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
            steps = np.geomspace(1, n - 1, min(4, n), dtype=int).tolist()
        else:
            steps = [s for s in steps if s < n]

        line_colors = plt.cm.nipy_spectral(np.linspace(0.1, 0.9, len(steps)))
        for i, num in enumerate(steps):
            plt.plot(
                r / w0,
                phr_f[num],
                c=line_colors[i],
                alpha=0.85,
                label=r"$\phi_R$ step %d" % num,
            )
        plt.plot(r / w0, ftf_t[:, fl], c="k", linestyle="--", linewidth=3, label=r"$\phi_T(z=f)$")
        plt.xlim((0, 0.2))
        plt.ylabel(r"$\phi$")
        plt.xlabel(r"$r/w_0$")
        plt.legend()
        plt.title("Focal-plane Amplitude over Iterations")
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
                print("I_F step: %d" % i)
                plt.plot(r / w0, it_f, label=r"$I_T(z=f)$")
                plt.plot(r / w0, ref, label=r"$I_R(z=f)$")
                plt.title("Step: %d" % i)
                plt.xlim((0, 0.2))
                plt.ylabel(r"$\phi$")
                plt.xlabel(r"$r/w_0$")
                plt.legend()
                plt.tight_layout()
                plt.savefig(path + "/I_f/I_f_%d" % i, dpi=100)
                plt.close()


def plot_I_f_over_iterations(path, ir_f, r, w0, it_f, steps=None, fontsize=30):
    """Overlay focal-plane intensity at selected iterations on a single plot.

    When the output path contains the string ``"int"``, the target intensity is
    normalised to its peak value before plotting. Saves to
    ``<path>/I_f_over_iterations.pdf``.

    Parameters
    ----------
    path : str
        Output directory where the PDF will be saved. If ``"int"`` appears in
        the path, ``it_f`` is peak-normalised.
    ir_f : np.ndarray, shape (iterations, N)
        Reflected field intensity at the focal plane for each iteration.
    r : np.ndarray, shape (N,)
        Radial coordinate array (metres).
    w0 : float
        Beam waist radius used to normalise the radial axis (metres).
    it_f : np.ndarray, shape (N,)
        Target intensity profile at the focal plane.
    steps : list of int or None, optional
        Specific iteration indices to include. If ``None``, up to 4 indices
        are selected via ``np.geomspace``. Default is ``None``.
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
            steps = np.geomspace(1, n - 1, min(4, n), dtype=int).tolist()
        else:
            steps = [s for s in steps if s < n]

        target_f = it_f / np.max(it_f) if "int" in path else it_f
        line_colors = plt.cm.nipy_spectral(np.linspace(0.1, 0.9, len(steps)))
        for i, num in enumerate(steps):
            plt.plot(
                r / w0,
                ir_f[num],
                c=line_colors[i],
                alpha=0.85,
                label=r"$I_R$ step %d" % num,
            )
        plt.plot(r / w0, target_f, c="k", linestyle="--", linewidth=3, label=r"$I_T(z=f)$")
        plt.xlim((0, 0.2))
        plt.ylabel("Focal-plane Intensity")
        plt.xlabel(r"$r/w_0$")
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        plt.legend()
        plt.title("Focal-plane Intensity over Iterations")
        plt.tight_layout()
        plt.savefig(path + "/I_f_over_iterations.pdf")
        plt.close()


# In[6]:


def plot_I_OA(path, ir_oa, z, f, it, steps, fontsize=30):
    """Save individual on-axis intensity comparisons at regular step intervals.

    For every ``steps``-th iteration, writes a figure comparing the reflected
    on-axis intensity ``I_R`` to the target ``I_T`` as a function of
    propagation distance ``z - f``, saved to ``<path>/I_OA/I_oa_<step>``.

    Parameters
    ----------
    path : str
        Output directory; images are written to the ``I_OA/`` sub-directory.
    ir_oa : np.ndarray, shape (iterations, M)
        Reflected on-axis intensity for each optimisation iteration.
    z : np.ndarray, shape (M,)
        Propagation axis coordinate array (metres).
    f : float
        Focal length; used to shift the axis so the focus is at the origin
        (metres).
    it : np.ndarray, shape (M,)
        Target on-axis intensity profile.
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
        for i, ref in enumerate(ir_oa):
            if i % steps == 0:
                print("Int_OA step: %d" % i)
                plt.plot((z - f) * 100, it, label=r"$I_T$")
                plt.plot((z - f) * 100, ref, label=r"$I_R$")
                plt.title("Step: %d" % i)
                plt.ylabel("On-axis Intensity (arb. units)")
                plt.xlabel(r"$z-f$ (cm)")
                plt.legend()
                plt.tight_layout()
                plt.savefig(path + "/I_OA/I_oa_%d" % i, dpi=100)
                plt.close()


def plot_I_OA_over_iterations(path, ir_oa, z, f, it, steps=None, fontsize=30):
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
        are selected via ``np.geomspace``. Default is ``None``.
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
            steps = np.geomspace(1, n - 1, min(4, n), dtype=int).tolist()
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
                label=r"$I_R$ step %d" % num,
            )
        plt.plot((z - f) * 100, target_oa, c="k", linestyle="--", linewidth=3, label=r"$I_T$")
        plt.ylabel("On-axis Intensity (arb. units)")
        plt.xlabel(r"$z-f$ (cm)")
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        plt.legend()
        plt.title("Reflected Intensity over Iterations")
        plt.tight_layout()
        plt.savefig(path + "/I_OA_over_iterations.pdf")
        plt.close()


def plot_phi_OA_over_iterations(path, phr_oa, z, f, ftf_t, steps=None, fontsize=30):
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
        are selected via ``np.geomspace``. Default is ``None``.
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
            steps = np.geomspace(1, n - 1, min(4, n), dtype=int).tolist()
        else:
            steps = [s for s in steps if s < n]

        line_colors = plt.cm.nipy_spectral(np.linspace(0.1, 0.9, len(steps)))
        for i, num in enumerate(steps):
            plt.plot(
                (z - f) * 100,
                phr_oa[num],
                c=line_colors[i],
                alpha=0.85,
                label=r"$\phi_R$ step %d" % num,
            )
        plt.plot((z - f) * 100, ftf_t[0, :], c="k", linestyle="--", linewidth=3, label=r"$\phi_T$")
        plt.ylabel("On-axis Amplitude")
        plt.xlabel(r"$z-f$ (cm)")
        plt.legend()
        plt.title("On-axis Amplitude over Iterations")
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
                print("PRPT_OA step: %d" % i)
                plt.plot((z - f) * 100, ftf_t[0, :], label=r"$\phi_T$")
                plt.plot((z - f) * 100, ref, label=r"$\phi_R$")
                plt.title("Step: %d" % i)
                plt.ylabel("On-axis Amplitude")
                plt.xlabel(r"$z-f$ (cm)")
                plt.legend()
                plt.tight_layout()
                plt.savefig(path + "/pr_oa/pr_oa_%d" % i, dpi=100)
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
                print("G step: %d" % i)
                plt.scatter(r / w0, dzm, label="df/dz")
                plt.scatter(r / w0, da, label="Zernike Fit on df/dz")
                plt.title("Step: %d" % i)
                plt.ylabel(r"$\delta z/\lambda$")
                plt.xlabel(r"$r/w_0$")
                plt.axhline(0, linestyle="--", c="k")
                if max(norm_r) > 2:
                    plt.xlim((0, 2))
                else:
                    plt.xlim(xmin=0)
                plt.legend()
                plt.tight_layout()
                plt.savefig(path + "/g/g_%d" % (i), dpi=100)
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
                print("Psi step: %d" % i)
                plt.plot(r / w0, np.real(ref))
                plt.title("Step: %d" % i)
                plt.ylabel(r"$\psi$")
                plt.xlabel(r"$r/w_0$")
                plt.tight_layout()
                plt.savefig(path + "/psi/psi_%d" % i, dpi=100)
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
                print("S step: %d" % i)
                plt.plot(z / f, np.real(ref))
                plt.title("Step: %d" % i)
                plt.ylabel(r"$S$")
                plt.xlabel(r"$z/f$")
                plt.tight_layout()
                plt.savefig(path + "/S/S_%d" % i, dpi=100)
                plt.close()


def plot_step0_intensity(path, ir_oa, ir_f, z, f, r, w0, It, fl, fontsize=30):
    """Two-panel figure (stacked vertically) of I_OA and I_f at step 0.

    Saves to ``<path>/step0_intensity.pdf``.
    """
    with plt.rc_context({"font.size": fontsize}):
        fig, (ax_oa, ax_f) = plt.subplots(2, 1, figsize=(16, 18))

        ax_oa.plot((z - f) * 100, It[0, :], label=r"$I_T$")
        ax_oa.plot((z - f) * 100, ir_oa[0], label=r"$I_R$ step 0")
        ax_oa.set_ylabel("On-axis Intensity (arb. units)")
        ax_oa.set_xlabel(r"$z-f$ (cm)")
        ax_oa.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax_oa.set_title(r"$I_{OA}$ at Step 0")
        ax_oa.legend()

        ax_f.plot(r / w0, It[:, fl], label=r"$I_T(z=f)$")
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

        ax_f.plot(r / w0, ftf_t[:, fl], label=r"$\phi_T(z=f)$")
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
                print(f"Plotting step: {i}")

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

                fig.suptitle(f"Step: {i}")
                plt.subplots_adjust(left=0.1, right=0.97, top=0.88, bottom=0.12)
                plt.savefig(path + "/psi_S/psi_S_%d" % i, dpi=100)
                plt.close()


def plot_gradient(path, gradient, r, w0, crop_rmax=2, linthresh=10, linscale=1.0, fontsize=26):
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

        def _draw(data, n_cols, r_hi, x_fmt, use_symlognorm):
            vabs = np.max(np.abs(data))
            if use_symlognorm:
                norm = SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=-vabs, vmax=vabs)
                plt.imshow(data, origin="lower", aspect="auto", cmap="RdBu_r", norm=norm)
            else:
                plt.imshow(data, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vabs, vmax=vabs)
            x_ticks = np.linspace(0, n_cols, 6)
            x_labels = np.linspace(0, r_hi, 6)
            y_ticks = np.linspace(0, n_iters, 11)
            y_labels = np.linspace(0, n_iters, 11)
            plt.xticks(x_ticks, [x_fmt.format(x) for x in x_labels])
            plt.yticks(y_ticks, [f"{y:.0f}" for y in y_labels])
            plt.xlabel(r"$r/w_0$")
            plt.ylabel("Iterations")
            plt.colorbar()
            plt.tight_layout()

        _draw(gradient, n_r, rmax, "{:.0f}", use_symlognorm=False)
        plt.savefig(path + "/gradient_full.pdf")
        plt.close()

        _draw(gradient, n_r, rmax, "{:.0f}", use_symlognorm=True)
        plt.savefig(path + "/gradient_full_symlognorm.pdf")
        plt.close()

        if rmax > crop_rmax:
            crop_idx = int(round(crop_rmax / rmax * n_r))
            cropped = gradient[:, :crop_idx]

            _draw(cropped, crop_idx, crop_rmax, "{:.1f}", use_symlognorm=False)
            plt.savefig(path + "/gradient_zoom.pdf")
            plt.close()

            _draw(cropped, crop_idx, crop_rmax, "{:.1f}", use_symlognorm=True)
            plt.savefig(path + "/gradient_zoom_symlognorm.pdf")
            plt.close()
