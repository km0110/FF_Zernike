import sys
import os
from gaussian_beam import *
from adjoint import *
from plot import *
import time
import matplotlib.pyplot as plt
import shutil

if len(sys.argv) != 2:
    print("Usage: python main.py <path to parameter file>")
    sys.exit(1)


path = sys.argv[1]
params = initialize_variables_from_file(path)
dirname = (path.split("/")[-1])[:-4]
path = "runs/" + dirname
if os.path.exists(path):
    shutil.rmtree(path)
os.makedirs(path)  # Calculate all initial parameters from the input file:
subdir = ["phr_f", "pr_oa", "g"]
for s in subdir:
    os.makedirs(path + "/" + s)

# Lens parameters
f = params["f"]  # focal length of lens
ell = 0  # order of Bessel J function
mir_i = params["mir_i"]
conic = params["conic"]

# Beam Parameters
wl = params["wl"]  # wavelength
k0 = 2 * np.pi / wl  # k=2*pi/wavelength
fn = params["fn"]  # f#
w0 = f / (2 * fn)  # initial spot size
rmax = params["r_maxf"] * w0  # the max value of r
nr = params["nr"]  # number of points in r
zR = np.pi * w0**2 / wl  # Initial rayleigh length
wR = 2 / np.pi * fn * wl  # Diffraction limited spot size

# Distance parameters
dz = params["dz"]
nzf = params["nzf"]
zF = np.linspace(f - dz, f + dz, nzf)
fl = len(zF) // 2  # focal length index

# Guess parameters
OoM = math.floor(math.log(w0**2, 10))  # order of magnitude of w0^2
w_guess = 10**OoM  # guess for the spot size
steps = params["steps"]  # max number of steps algorithm is allowed
lr = params["lr"]  # learning rate
tol = params["tol"]  # tolerance
delta = params["delta"]  # weighting function in r

# Hankel
[r, k] = DHTarrays(nr, rmax, ell)
[T_R2K, T_K2R] = DHTtransformMatrix(nr, rmax, ell)

# Initial propagation
phi_I = 1j * np.exp(-((r / w0) ** 2))
zm_t = create_starting_optic(r, 2 * f)  # target mirror
ftf_t = forward_propagation(w0, r, k, phi_I, T_R2K, T_K2R, zF, zm_t)
zm_r = create_starting_optic(r, 2 * mir_i, k=conic)  # initial mirror

# Running the main algorithm once
start_time = time.time()
ftf_r = forward_propagation(w0, r, k, phi_I, T_R2K, T_K2R, zF, zm_r)
psi_hat = adjoint(
    r, 2 * wR, k, k0, zF, f - dz, f + dz, ftf_r, ftf_t, T_R2K, T_K2R, delta
)
psi = T_K2R @ psi_hat
grad = np.real(gradient(k0, phi_I, psi, zm_r))
