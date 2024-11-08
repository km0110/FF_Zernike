import numpy as np
from plot import *
from adjoint import FOM, calc_mu, initialize_variables_from_file
import sys
from datetime import timedelta
import time
import shutil
import os

# read path to data from command line
if len(sys.argv) != 3:
    print("Usage: python plot_from_data.py <path to parameter file>")
    sys.exit(1)

path = sys.argv[1]
params = initialize_variables_from_file(sys.argv[1])
dirname = (path.split("/")[-1])[:-4]
path = path.split("/")[0] + "/runs/main_" + dirname
subdir = ["phr_f", "pr_oa", "g", "s", "mirror"]
for s in subdir:
    os.makedirs(path + "/" + s)

data_names = [
    "phi_r_OA.csv",
    "phi_r_focus.csv",
    "gradient.csv",
    "mirror.csv",
    "source.csv",
    "zernike.csv",
]

f_targ = params["f_targ"]  # focal length of lens
wl = params["wl"]  # wavelength
lr = params["lr"]  # learning rate
n = params["n"]  # number of zernike terms
dz = params["dz"]
nzf = params["nzf"]
zi = f_targ - dz
L = params["L"]
zf = f_targ + L + dz
zF = np.linspace(zi, zf, nzf)

# find index of where zF=0
fl = np.where(zF == f_targ)[0]


# read in each of the data
filenames = [path + "/" + s for s in data_names]

phr_oa, phr_f, gr, mr, src, zern = [np.genfromtxt(f, delimiter=",") for f in filenames]
pr2, rt, pt2 = np.genfromtxt(path + "/data.csv", delimiter=",", unpack=True)

i = len(pr2)
# calculate figure of merit and mu
figom = FOM(pr2, rt, pt2)
mu_val = calc_mu(rt, pt2)

norm_r = r / w0

# find index of where zF=0


print("START PLOTTING")
pstart = time.time()
# Plotting overall change
plot_total_change(path, norm_r, zern, lr, wl, i, n)
plot_max_grad(path, gr)
plot_fom_mu(path, figom, mu_val)
plot_integrals(path, pr2, rt, pt2)
plot_mirror_comp(path, norm_r, zm_t, mr, i)
plot_mirror_comp_each(path, norm_r, zm_t, mr)
plot_zernikes(path, zern, rmax, f_targ, n)
pplots = time.time()
print("Plotting Time:" + str(timedelta(seconds=pplots - pstart)))

# Plotting each iteration phi_r and phi_t, the gradient and the source
plot_phi_t_phi_r_f(path, phr_f, norm_r, ftf_t, fl)
prft = time.time()
print("PRPT_F Time:" + str(timedelta(seconds=prft - pplots)))

plot_phi_t_phi_r_OA(path, phr_oa, zF, f_targ, ftf_t)
proat = time.time()
print("PRPT_OA Time:" + str(timedelta(seconds=proat - prft)))

plot_da_vs_grad(path, lr, gr, norm_r, wl, zern, n)
pdzt = time.time()
print("G Time:" + str(timedelta(seconds=pdzt - proat)))
plot_mirror_comp_each(path, norm_r, zm_t, mr)

# plot_s(path, src, zF, f_targ)
# ps = time.time()
# print("S Time:" + str(timedelta(seconds=ps - pdzt)))

pend = time.time()

print("Total Time:" + str(timedelta(seconds=pend - pstart)))
