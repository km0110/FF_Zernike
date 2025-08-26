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
    "f_targ": 0.5,
    "f_init": 0.5,
    "ell": 0,
    "wl": 1e-6,
    "f_num": 25,
    "conic": -1,
    "nr": 1000,
    "r_maxf": 5,
    "nzf": 501,
    "dz": 1e-1,
    "steps": 10000,
    "lr": 1e-10,
    "delta": 1,
    "tol": 1e-16,
    "sg": 0,
    "order": 0,
    "n": 3,
    "L": 5e-2,
    "skip": 10,
    "stop": 1000001,
    "temp": 10,
    "c": 1,
    "factor": 0.5,
    "mu_tol": 1e-4,
    "sigma": [5e-3, 1e-2, 2e-2, 3e-2],
}


def check_spot_size(param):
    # This is to check that the spot size is greater than the diffraction limited spot size, wR
    wR = 2 / np.pi * param["f_num"] * param["wl"]
    w0 = param["f_init"] / (2 * param["f_num"])
    return w0 > wR


def create_filename(file_dir, param):
    filename = "_".join([f"{k}_{v}" for k, v in param.items()]) + ".txt"
    if param["delta"]:
        return file_dir + "/delta_" + filename
    else:
        return file_dir + "/exp_" + filename


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
    # if check_spot_size(params):
    #     dirname = sys.argv[1]
    #     os.makedirs(dirname, exist_ok=True)
    #     create_run_files(dirname, params)
    # else:
    #     print("Spot size is too small for the given parameters")
    dirname = sys.argv[1]
    os.makedirs(dirname, exist_ok=True)
    create_run_files(dirname, params)


if __name__ == "__main__":
    main()
