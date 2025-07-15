#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from adjoint import create_initial_coeff, calc_Z

font = {"size": 26}

plt.rc("font", **font)
plt.rcParams["figure.figsize"] = (16, 9)
plt.rcParams["lines.linewidth"] = 3


def plot_total_change(path, r, w0, gr, lr, wl, steps, n, xlim=(0, 2), spikes=None):
    Z_fit = calc_Z(r / np.max(r), n)
    norm_r = r / w0
    dfda = Z_fit @ gr.T
    plt.scatter(
        norm_r, -1 * np.sum(dfda, axis=1) * lr / wl, label="After %d Steps" % (steps)
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
def plot_fom_mu(path, fom, mu):
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


def plot_integrals(path, prsq, prpt, ptsq):
    plt.plot(prsq * ptsq, label=r"$\langle\phi_R^2\rangle \langle\phi_T^2\rangle$")
    plt.plot(prpt * prpt, label=r"$\langle\phi_R\phi_T\rangle^2$")
    plt.xlabel("Iterations")
    plt.legend()
    plt.savefig(path + "/prpt")
    plt.close()


def plot_max_grad(path, gr):
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


def plot_mirror_comp(path, r, w0, z_m, mr):
    norm_r = r / w0
    plt.plot(norm_r, z_m, label="Target Mirror")
    plt.plot(norm_r, mr[0], label="Initial Mirror")
    plt.plot(norm_r, mr[-1], linestyle="--", label="Final Mirror")

    # me=np.loadtxt('exp_mr.csv')
    # plt.plot(r,me,label='exp',linestyle='--',linewidth=3)
    if max(norm_r) > 2:
        plt.xlim((0, 2))
        plt.ylim((0, 3e-5))
    else:
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)
    plt.xlabel(r"$r/w_0$")
    plt.ylabel("z (m)")
    plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    plt.legend()

    plt.tight_layout()
    plt.savefig(path + "/mirror_comp.png")
    plt.close()


# plot mirror at each step and compare to target and save to file
def plot_mirror_comp_each(path, r, w0, zm_t, mr, steps, xlim=(0, 2), ylim=(0, 3e-5)):
    norm_r = r / w0

    for i, ref in enumerate(mr):
        if i % steps == 0:
            print("Mirror step: %d" % i)
            plt.figure(figsize=(11.52, 6.56), dpi=100)
            plt.plot(norm_r, zm_t, label="Target Mirror")
            plt.plot(norm_r, ref, label="Mirror at step %d" % i)
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
def plot_mirror_all(path, r, w0, mr, xlim=(0, 2), ylim=(0, 3e-5)):
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


def plot_zernikes(path, zern, rmax, f_targ, n):
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


def plot_phi_t_phi_r_f(path, phr_f, r, w0, ftf_t, fl, steps):
    for i, ref in enumerate(phr_f):
        if i % steps == 0:
            print("PRPT_F step: %d" % i)
            # plt.figure(figsize=(11.52, 6.56), dpi=100)
            plt.plot(r / w0, ftf_t[:, fl], label=r"$\phi_T(z=f)$")

            plt.plot(r / w0, ref, label=r"$\phi_R(z=f)$")
            plt.title("Step: %d" % i)
            plt.xlim((0, 1))
            plt.ylabel(r"$\phi$")
            plt.xlabel(r"$r/w_0$")
            plt.legend()
            #     plt.show()
            plt.tight_layout()
            plt.savefig(path + "/phr_f/phr_f_%d" % i, dpi=100)
            plt.close()


def plot_I_f(path, ir_f, r, w0, it_f, steps):
    for i, ref in enumerate(ir_f):
        if i % steps == 0:
            print("I_F step: %d" % i)
            # plt.figure(figsize=(11.52, 6.56), dpi=100)
            plt.plot(r / w0, it_f, label=r"$I_T(z=f)$")

            plt.plot(r / w0, ref, label=r"$I_R(z=f)$")
            plt.title("Step: %d" % i)
            plt.xlim((0, 1))
            plt.ylabel(r"$\phi$")
            plt.xlabel(r"$r/w_0$")
            plt.legend()
            #     plt.show()
            plt.tight_layout()
            plt.savefig(path + "/I_f/I_f_%d" % i, dpi=100)
            plt.close()


# In[6]:


def plot_I_OA(path, ir_oa, z, f, it, steps):
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


def plot_phi_t_phi_r_OA(path, phr_oa, z, f, ftf_t, steps):
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


def plot_da_vs_grad(path, gr, r, w0, zern, n, skip):
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


def plot_psi(path, psi, r, w0, steps):
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


def plot_source(path, S, z, f, steps):
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
