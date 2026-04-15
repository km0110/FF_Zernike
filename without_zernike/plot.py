#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np

# from adjoint import create_initial_coeff, calc_Z

font = {"size": 26}

plt.rc("font", **font)
plt.rcParams["figure.figsize"] = (16, 9)
plt.rcParams["lines.linewidth"] = 3


def spikes(mirror):
    ddfmr = np.diff(mirror, n=2)
    return np.where(ddfmr < -4.5e-7)[0] + 1


def shift(mirror, spikes):
    new_mir = np.copy(mirror)
    n = len(spikes)
    spikes = np.append(spikes, len(mirror))
    for i in range(n):
        start = spikes[i]
        end = spikes[i + 1]
        new_mir[start:end] -= (i + 1) * 5e-7
    return new_mir


def plot_total_change(path, r, w0, gr, lr, wl, steps, xlim=(0, 2), spikes=None, fontsize=26):
    with plt.rc_context({"font.size": fontsize}):
        # Z_fit = calc_Z(r / np.max(r), n)
        norm_r = r / w0
        # dfda = Z_fit @ gr.T
        plt.scatter(
            norm_r, -1 * np.sum(gr, axis=0) * lr / wl, label="After %d Steps" % (steps)
        )
        # plt.xlim((0,2*w0))
        plt.title("Total change in the mirror")
        plt.xlabel(r"$r/w_0$")
        plt.ylabel(r"$\sum\delta z/\lambda$")
        plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
        # plt.axvline(w0,label='$w_0$',linestyle='--',c='k')
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
        plt.savefig(path + "/total_change")
        plt.close()
        # plt.savefig('f-0_01_total_change_%d_steps_full'%(i))


# In[3]:
def plot_fom_mu(path, fom, mu, fontsize=26):
    with plt.rc_context({"font.size": fontsize}):
        # Sample data
        # TODO: change the ylim to [0,1]
        # Create the first plot with the left y-axis
        fig, ax1 = plt.subplots()

        color = "tab:red"
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel(r"$\mu$", color=color)
        ax1.plot(mu, color=color)
        ax1.tick_params(axis="y", labelcolor=color)

        # Create the second plot with the right y-axis
        ax2 = ax1.twinx()
        color = "tab:blue"
        ax2.set_ylabel("Figure of Merit", color=color)
        ax2.plot(fom / fom[0], color=color)
        ax2.tick_params(axis="y", labelcolor=color)
        # plt.ylim(0, 1)
        # Show the plot
        plt.savefig(path + "/fom_mu")
        plt.close()


def plot_percent_change_mu(path, mu, fontsize=26):
    with plt.rc_context({"font.size": fontsize}):
        percent_change = (np.diff(mu) / mu[:-1]) * 100
        iterations = range(len(percent_change)) + 1
        plt.plot(iterations, percent_change)
        plt.savefig(path + "/pc_mu")
        plt.close()


def plot_integrals(path, prsq, prpt, ptsq, fontsize=26):
    with plt.rc_context({"font.size": fontsize}):
        plt.plot(prsq * ptsq, label=r"$\langle\phi_R^2\rangle \langle\phi_T^2\rangle$")
        plt.plot(prpt * prpt, label=r"$\langle\phi_R\phi_T\rangle^2$")
        plt.xlabel("Iterations")
        plt.legend()
        plt.savefig(path + "/prpt")
        plt.close()


def plot_max_grad(path, gr, fontsize=26):
    with plt.rc_context({"font.size": fontsize}):
        plt.plot(np.max(np.real(gr), axis=1))
        # print(np.min(np.max(np.abs(gr),axis=1)))
        plt.xlabel("Iteration")
        plt.ylabel(r"$\max [\frac{dF}{dz_m}]$")
        # plt.axvline(30,color='k',linestyle='--',linewidth=2,label='Step: 30')
        # plt.axvline(300,color='k',linestyle='--',linewidth=2)
        plt.savefig(path + "/max_gr")
        plt.close()
        # plt.savefig('max_grad_0-30')

        # plt.show()


# # Compare $\phi_T$ to $\phi_R$ for each step

# In[4]:


# # Show Target, Initial, and Final Mirror

# In[8]:


def plot_mirror_comp(path, r, w0, z_m, mr, steps=None, xlim=(0, 2), ylim=(0, 3e-5), do_shift=False, fontsize=26):
    with plt.rc_context({"font.size": fontsize}):
        norm_r = r / w0
        plt.plot(norm_r, z_m, label="Target Mirror")
        plt.plot(norm_r, mr[0], label="Initial Mirror")

        sp = spikes(mr[-1])
        if do_shift:
            final = shift(mr[-1], sp)
            name = "/mirror_comp_shifted"
        else:
            final = mr[-1]
            name = "/mirror_comp"

        label = "After %d Steps" % steps if steps is not None else "Final Mirror"
        plt.plot(norm_r, final, linestyle="--", linewidth=4, label=label)

        for s in r[sp]:
            plt.axvline(s / w0, linestyle="--", c="k", linewidth=2)

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
        plt.savefig(path + name)
        plt.close()


# plot mirror at each step and compare to target and save to file
def plot_mirror_comp_each(path, r, w0, zm_t, mr, steps, xlim=(0, 2), ylim=(0, 3e-5), fontsize=26):
    with plt.rc_context({"font.size": fontsize}):
        norm_r = r / w0

        for i, ref in enumerate(mr):
            if i % steps == 0:
                print("Mirror step: %d" % i)
                plt.figure(figsize=(11.52, 6.56), dpi=100)
                plt.plot(norm_r, zm_t, label="Target Mirror")
                plt.plot(norm_r, ref - ref[0], label="Mirror at step %d" % i)
                # keep xlim and ylim across all plots by setting it based on min and max x and y values in mr
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


# plot all of the mirror steps on one plot
def plot_mirror_all(path, r, w0, mr, xlim=(0, 2), ylim=(0, 3e-5), fontsize=26):
    with plt.rc_context({"font.size": fontsize}):
        norm_r = r / w0

        for i, ref in enumerate(mr):
            if i % 50 == 0:
                print("Mirror step: %d" % i)
            plt.plot(norm_r, ref, label="Mirror at step %d" % i)
            # if max(norm_r) > 2:
            #     plt.xlim(xlim)
            #     plt.ylim(ylim)
            # else:
            plt.xlim(xmin=0)
            plt.ylim(ymin=0)
            plt.legend()
            plt.xlabel(r"$r/w_0$")
            plt.ylabel("z (m)")
            plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
            plt.tight_layout()
        plt.savefig(path + "/mirror/mirror_all", dpi=100)
        plt.close()


def plot_zernikes(path, zern, rmax, f_targ, n, fontsize=26):
    with plt.rc_context({"font.size": fontsize}):
        # each column should be a separate plot
        for i in range(zern.shape[1]):
            plt.plot(zern[:, i], label=f"Zernike Z{2*i},0")
        plt.xlabel("Iteration")
        plt.ylabel("Coefficient")

        # ideal coefficients for the target mirror
        # zm_t = create_initial_coeff(rmax, 2 * f_targ, n)

        # create horizontal lines for each y value in zm_t
        # for i, z in enumerate(zm_t):
        #     # different colors for each line
        #     plt.axhline(
        #         z,
        #         linestyle="--",
        #         c=plt.cm.viridis(i / len(zm_t)),
        #         label="Target Zernike Z%i0 = %.2e" % (i, z),
        #     )
        plt.legend()

        plt.tight_layout()
        plt.savefig(path + "/zernikes.png")
        plt.close()


def plot_phi_f(path, phr_f, r, w0, ftf_t, fl, steps, fontsize=26):
    with plt.rc_context({"font.size": fontsize}):
        for i, ref in enumerate(phr_f):
            if i % steps == 0:
                print("PRPT_F step: %d" % i)
                # plt.figure(figsize=(11.52, 6.56), dpi=100)
                plt.plot(r / w0, ftf_t[:, fl], label=r"$\phi_T(z=f)$")

                plt.plot(r / w0, ref, label=r"$\phi_R(z=f)$")
                plt.title("Step: %d" % i)
                plt.xlim((0, 0.2))
                plt.ylabel(r"$\phi$")
                plt.xlabel(r"$r/w_0$")
                plt.legend()
                #     plt.show()
                plt.tight_layout()
                plt.savefig(path + "/pr_f/pr_f_%d" % i, dpi=100)
                plt.close()


def plot_phi_f_over_iterations(path, phr_f, r, w0, ftf_t, fl, steps=None, fontsize=26):
    """Overlay focal-plane amplitude at selected iterations on a single plot."""
    with plt.rc_context({"font.size": fontsize}):
        n = len(phr_f)
        if steps is None:
            steps = np.linspace(0, n - 1, min(6, n), dtype=int).tolist()
        else:
            steps = [s for s in steps if s < n]

        line_colors = plt.cm.turbo(np.linspace(0.08, 0.92, len(steps)))
        plt.plot(r / w0, ftf_t[:, fl], c="k", label=r"$\phi_T(z=f)$")
        for i, num in enumerate(steps):
            plt.plot(
                r / w0,
                phr_f[num],
                alpha=0.7,
                c=line_colors[i],
                label=r"$\phi_R$ step %d" % num,
            )
        plt.xlim((0, 0.2))
        plt.ylabel(r"$\phi$")
        plt.xlabel(r"$r/w_0$")
        plt.legend()
        plt.title("Focal-plane Amplitude over Iterations")
        plt.tight_layout()
        plt.savefig(path + "/phi_f_over_iterations")
        plt.close()


def plot_I_f(path, ir_f, r, w0, it_f, steps, fontsize=26):
    with plt.rc_context({"font.size": fontsize}):
        for i, ref in enumerate(ir_f):
            if i % steps == 0:
                print("I_F step: %d" % i)
                # plt.figure(figsize=(11.52, 6.56), dpi=100)
                plt.plot(r / w0, it_f, label=r"$I_T(z=f)$")

                plt.plot(r / w0, ref, label=r"$I_R(z=f)$")
                plt.title("Step: %d" % i)
                plt.xlim((0, 0.2))
                plt.ylabel(r"$\phi$")
                plt.xlabel(r"$r/w_0$")
                plt.legend()
                #     plt.show()
                plt.tight_layout()
                plt.savefig(path + "/I_f/I_f_%d" % i, dpi=100)
                plt.close()


def plot_I_f_over_iterations(path, ir_f, r, w0, it_f, steps=None, fontsize=26):
    """Overlay focal-plane intensity at selected iterations on a single plot."""
    with plt.rc_context({"font.size": fontsize}):
        n = len(ir_f)
        if steps is None:
            steps = np.linspace(0, n - 1, min(6, n), dtype=int).tolist()
        else:
            steps = [s for s in steps if s < n]

        line_colors = plt.cm.turbo(np.linspace(0.08, 0.92, len(steps)))
        plt.plot(r / w0, it_f, c="k", label=r"$I_T(z=f)$")
        for i, num in enumerate(steps):
            plt.plot(
                r / w0,
                ir_f[num],
                alpha=0.7,
                c=line_colors[i],
                label=r"$I_R$ step %d" % num,
            )
        plt.xlim((0, 0.2))
        plt.ylabel("Focal-plane Intensity")
        plt.xlabel(r"$r/w_0$")
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        plt.legend()
        plt.title("Focal-plane Intensity over Iterations")
        plt.tight_layout()
        plt.savefig(path + "/I_f_over_iterations")
        plt.close()


# In[6]:


def plot_I_OA(path, ir_oa, z, f, it, steps, fontsize=26):
    with plt.rc_context({"font.size": fontsize}):
        for i, ref in enumerate(ir_oa):
            if i % steps == 0:
                print("Int_OA step: %d" % i)
                # plt.figure(figsize=(16, 6.56), dpi=100)
                plt.plot((z - f) * 100, it, label=r"$I_T$")

                plt.plot((z - f) * 100, ref, label=r"$I_R$")
                plt.title("Step: %d" % i)
                plt.ylabel("On-axis Intensity")
                plt.xlabel(r"$z-f$ (cm)")
                plt.legend()
                #     plt.show()
                plt.tight_layout()
                plt.savefig(path + "/I_OA/I_oa_%d" % i, dpi=100)
                plt.close()


def plot_I_OA_over_iterations(path, ir_oa, z, f, it, steps=None, fontsize=26):
    """Overlay on-axis intensity at selected iterations on a single plot."""
    with plt.rc_context({"font.size": fontsize}):
        n = len(ir_oa)
        if steps is None:
            steps = np.linspace(0, n - 1, min(6, n), dtype=int).tolist()
        else:
            steps = [s for s in steps if s < n]

        line_colors = plt.cm.turbo(np.linspace(0.08, 0.92, len(steps)))
        plt.plot((z - f) * 100, it, c="k", label=r"$I_T$")
        for i, num in enumerate(steps):
            plt.plot(
                (z - f) * 100,
                ir_oa[num],
                alpha=0.7,
                c=line_colors[i],
                label=r"$I_R$ step %d" % num,
            )
        plt.ylabel("On-axis Intensity")
        plt.xlabel(r"$z-f$ (cm)")
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        plt.legend()
        plt.title("Reflected Intensity over Iterations")
        plt.tight_layout()
        plt.savefig(path + "/I_OA_over_iterations")
        plt.close()


def plot_phi_OA_over_iterations(path, phr_oa, z, f, ftf_t, steps=None, fontsize=26):
    """Overlay on-axis amplitude at selected iterations on a single plot."""
    with plt.rc_context({"font.size": fontsize}):
        n = len(phr_oa)
        if steps is None:
            steps = np.linspace(0, n - 1, min(6, n), dtype=int).tolist()
        else:
            steps = [s for s in steps if s < n]

        line_colors = plt.cm.turbo(np.linspace(0.08, 0.92, len(steps)))
        plt.plot((z - f) * 100, ftf_t[0, :], c="k", label=r"$\phi_T$")
        for i, num in enumerate(steps):
            plt.plot(
                (z - f) * 100,
                phr_oa[num],
                alpha=0.7,
                c=line_colors[i],
                label=r"$\phi_R$ step %d" % num,
            )
        plt.ylabel("On-axis Amplitude")
        plt.xlabel(r"$z-f$ (cm)")
        plt.legend()
        plt.title("On-axis Amplitude over Iterations")
        plt.tight_layout()
        plt.savefig(path + "/phi_OA_over_iterations")
        plt.close()


def plot_phi_OA(path, phr_oa, z, f, ftf_t, steps, fontsize=26):
    with plt.rc_context({"font.size": fontsize}):
        for i, ref in enumerate(phr_oa):
            if i % steps == 0:
                print("PRPT_OA step: %d" % i)
                # plt.figure(figsize=(16, 6.56), dpi=100)
                plt.plot((z - f) * 100, ftf_t[0, :], label=r"$\phi_T$")

                plt.plot((z - f) * 100, ref, label=r"$\phi_R$")
                plt.title("Step: %d" % i)
                plt.ylabel("On-axis Amplitude")
                plt.xlabel(r"$z-f$ (cm)")
                plt.legend()
                #     plt.show()
                plt.tight_layout()
                plt.savefig(path + "/pr_oa/pr_oa_%d" % i, dpi=100)
                plt.close()


# # Show each step size

# In[37]:


def plot_da_vs_grad(path, gr, r, w0, zern, n, skip, fontsize=26):
    with plt.rc_context({"font.size": fontsize}):
        i = 0
        Z_fit = calc_Z(r / np.max(r), n)
        dfda = Z_fit @ zern.T
        norm_r = r / w0
        for dzm, da in zip(gr, dfda.T):
            if i % skip == 0:
                print("G step: %d" % i)
                # plt.figure(figsize=(11.52, 6.56), dpi=100)
                plt.scatter(r / w0, dzm, label="df/dz")
                plt.scatter(r / w0, da, label="Zernike Fit on df/dz")
                # plt.plot(r,np.real(ref),label='real')
                plt.title("Step: %d" % i)
                plt.ylabel(r"$\delta z/\lambda$ ")
                plt.xlabel(r"$r/w_0$")
                plt.axhline(0, linestyle="--", c="k")
                if max(norm_r) > 2:
                    plt.xlim((0, 2))
                else:
                    plt.xlim(xmin=0)
                # plt.ylim((np.min(zern) / wl, np.max(zern) / wl))
                #     plt.show()
                plt.legend()
                plt.tight_layout()
                plt.savefig(path + "/g/g_%d" % (i), dpi=100)

                plt.close()
            i += 1


def plot_psi(path, psi, r, w0, steps, fontsize=26):
    with plt.rc_context({"font.size": fontsize}):
        for i, ref in enumerate(psi):
            if i % steps == 0:
                print("Psi step: %d" % i)
                # plt.figure(figsize=(11.52, 6.56), dpi=100)
                plt.plot(r / w0, np.real(ref))
                plt.title("Step: %d" % i)
                plt.ylabel(r"$\psi$")
                plt.xlabel(r"$r/w_0$")
                #     plt.show()
                plt.tight_layout()
                plt.savefig(path + "/psi/psi_%d" % i, dpi=100)
                plt.close()


def plot_source(path, S, z, f, steps, fontsize=26):
    with plt.rc_context({"font.size": fontsize}):
        for i, ref in enumerate(S):
            if i % steps == 0:
                print("S step: %d" % i)
                # plt.figure(figsize=(11.52, 6.56), dpi=100)
                plt.plot(z / f, np.real(ref))
                plt.title("Step: %d" % i)
                plt.ylabel(r"$S$")
                plt.xlabel(r"$z/f$")
                #     plt.show()
                plt.tight_layout()
                plt.savefig(path + "/S/S_%d" % i, dpi=100)
                plt.close()


def plot_psi_and_source(path, psi, r, w0, S, z, f, skip, fontsize=26):
    with plt.rc_context({"font.size": fontsize}):
        # Ensure the combined directory exists

        for i, (psi_ref, S_ref) in enumerate(zip(psi, S)):
            if i % skip == 0:
                print(f"Plotting step: {i}")

                fig, axes = plt.subplots(
                    1, 2, figsize=(16, 9), dpi=100, gridspec_kw={"wspace": 0.25}
                )

                # Plot psi(r)
                axes[0].plot(z / f, np.real(S_ref), color="green")
                axes[0].set_title(r"$S(z)$")
                axes[0].set_xlabel(r"$z/f$")
                axes[0].set_ylabel(r"$S$")

                # Plot psi(r) on the right
                axes[1].plot(r / w0, np.real(psi_ref), color="blue")
                axes[1].set_title(r"$\psi(r)$")
                axes[1].set_xlabel(r"$r/w_0$")
                axes[1].set_ylabel(r"$\psi$")
                # Overall title
                fig.suptitle(f"Step: {i}")
                plt.subplots_adjust(left=0.1, right=0.97, top=0.88, bottom=0.12)
                # plt.tight_layout()  # Reserve space for suptitle
                plt.savefig(path + "/psi_S/psi_S_%d" % i, dpi=100)
                plt.close()
