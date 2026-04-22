#!/usr/bin/env python
# coding: utf-8


import numpy as np
from itertools import product
import os
import sys

if len(sys.argv) != 2:
    print("Usage: python create_params.py <name of parameters directory>")
    sys.exit(1)


# Lens parameters
params = {
    "lr": 1e-10,
    "f_targ": 0.05,
    "f_init": 0.045,
    "nr": 1000,
    "r_maxf": 5,
    "nzf": 501,
    "dz": 2e-2,
    "L": 0,
    "steps": 1000,
    "skip": 1,
    "f_num": 25,
    "ell": 0,
    "wl": 1e-6,
    "conic": -1,
    "delta": 1,
    "tol": 1e-14,
    "sg": 0,
    "order": 0,
    # "stop": 1000001,
    "temp": 10,
    "factor": 0.5,
}


def check_z_resolution(param):
    # check that there is enough longitudinal resolution
    wl = param["wl"]
    k0 = 2 * np.pi / wl
    nr = param["nr"]
    ell = param.get("ell", 0)
    w0 = param["f_init"] / (2 * param["f_num"])
    rmax = param["r_maxf"] * w0
    Zeroes = np.genfromtxt("BesselZeroes" + str(ell) + ".csv")
    kmax = Zeroes[nr - 1] / rmax
    L = param.get("L", 0)
    if isinstance(L, list):
        L = max(L)
    dz_step = (L + 2 * param["dz"]) / (param["nzf"] - 1)
    z_res = 4 * np.pi * k0 / kmax**2
    print(f"dz step: {dz_step:.3e}, z_res: {z_res:.3e}")
    assert dz_step < z_res, "Need more resolution in z, increase number of points"


def check_spot_size(param):
    # This is to check that the spot size is greater than the diffraction limited spot size, wR
    wR = 2 / np.pi * param["f_num"] * param["wl"]
    w0 = param["f_init"] / (2 * param["f_num"])
    print(f"w0: {w0:.3e}, wR: {wR:.3e}")
    assert w0 > wR, "Spot size is too small, adjust f number"


# Keys to include in filename (in order), with their abbreviations
FILENAME_KEYS = {
    "lr": "lr",
    "f_targ": "ft",
    "f_init": "fi",
    "nr": "nr",
    "r_maxf": "rmx",
    "nzf": "nzf",
    "dz": "dz",
    "L": "L",
    "steps": "st",
    "skip": "sk",
    "f_num": "fn",
}


def create_filename(file_dir, param):
    parts = [
        f"{abbrev}_{param[k]}" for k, abbrev in FILENAME_KEYS.items() if k in param
    ]
    filename = "_".join(parts) + ".txt"
    prefix = "d" if param["delta"] else "e"
    return os.path.join(file_dir, f"{prefix}_{filename}")


def create_parameter_file(file_path, variables):
    try:
        with open(file_path, "w") as file:
            for name, value in variables.items():
                file.write(f"{name} = {repr(value)}\n")
        print(f"Parameter file '{file_path}' created successfully.")
    except Exception as e:
        print(f"Error: {e}")


def create_run_files(file_dir, param):
    # Creates multiple files based on the input parameters
    # For example: lr=[1e-11,1e-12] would make 2 parameter files one with each learning rate
    # If you have multiple parameters with multiple options it would then make a different parameter file for each combination of parameters
    # For example if you have lr=[1e-11,1e-12] and dz=[1.5e-2,2e-2] then you would have 4 different parameter files for each possible combination
    values = list(param.values())
    lists = [value if isinstance(value, list) else [value] for value in values]
    combinations = product(*lists)
    run_list = []
    for combo in combinations:
        combined_dict = dict(zip(param.keys(), combo))
        c_fname = create_filename(file_dir, combined_dict)
        create_parameter_file(c_fname, combined_dict)

    print("COMPLETE")


def main():
    check_z_resolution(params)
    check_spot_size(params)
    dirname = sys.argv[1]
    os.makedirs(dirname, exist_ok=True)
    create_run_files(dirname, params)

if __name__ == "__main__":
    main()
