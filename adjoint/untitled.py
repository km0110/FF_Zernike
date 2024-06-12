def plot_all(path, phr_f, phr_oa, delta_zm, s, r_norm, z_norm, ftf_t, fl):
    for i in len(phr_f):
        if i % 50:
            print("Plot step: %d" % i)
    plot_phi_t_phi_r_f(path, phr_f, r, w0, ftf_t, fl)


def plot_phi_t_phi_r_f(path, phr_f, r_norm, ftf_t, fl, i):
    plt.figure(figsize=(11.52, 6.56), dpi=100)
    plt.plot(r / w0, np.real(ftf_t[:, fl]), label=r"$\phi_T(z=f)$")

    plt.plot(r / w0, np.real(ref), label=r"$\phi_R(z=f)$")
    plt.title("Step: %d" % i)
    plt.xlim((0, 1))
    plt.ylabel(r"$\phi$")
    plt.xlabel(r"$r/w_0$")
    plt.legend()
    #     plt.show()
    plt.tight_layout()
    plt.savefig(path + "/phr_f/phr_f_%d" % i, dpi=100)
    plt.close()


# In[6]:


def plot_phi_t_phi_r_OA(path, phr_oa, z_norm, ftf_t, i):
    plt.figure(figsize=(11.52, 6.56), dpi=100)
    plt.plot(z / f, np.real(ftf_t[0, :]), label=r"$\phi_T$")

    plt.plot(z / f, np.real(ref), label=r"$\phi_R$")
    plt.title("Step: %d" % i)
    plt.ylabel("On-axis amplitude")
    plt.xlabel(r"$z/f$")
    plt.legend()
    #     plt.show()
    plt.tight_layout()
    plt.savefig(path + "/pr_oa/pr_oa_%d" % i, dpi=100)
    plt.close()


# # Show each step size

# In[37]:


def plot_dz(path, dzm, r_norm, wl, i):
    plt.figure(figsize=(11.52, 6.56), dpi=100)
    plt.plot(r / w0, dzm, label="step %d" % i)
    # plt.plot(r,np.real(ref),label='real')
    plt.title("Step: %d" % i)

    plt.ylabel(r"$\delta z/\lambda$ ")
    plt.xlabel(r"$r/w_0$")
    plt.axhline(0, linestyle="--", c="k")
    plt.xlim((0, 2))
    plt.ylim((np.min(delta_zm) / wl, np.max(delta_zm) / wl))
    #     plt.show()
    plt.savefig(path + "/g/g_%d" % (i), dpi=100)
    plt.close()


def plot_s(path, s, z, f):
    for i, ref in enumerate(s):
        if i % 50 == 0:
            print("S step: %d" % i)
        plt.figure(figsize=(11.52, 6.56), dpi=100)
        plt.plot(z / f, np.real(ref))
        plt.title("Step: %d" % i)
        plt.ylabel(r"$S=\phi_R-\mu\phi_T$")
        plt.xlabel(r"$z/f$")
        #     plt.show()
        plt.tight_layout()
        plt.savefig(path + "/s/s_%d" % i, dpi=100)
        plt.close()
