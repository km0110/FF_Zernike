#!/usr/bin/env python
"""
Sweep over Zernike mode count j, running lr_scan.py (learning-rate sweep) for
each j value.

Shared across experiment directories -- lives in common/, alongside
lr_scan.py which it calls once per j. Run from inside the experiment
directory (same as lr_scan.py) so the main script resolves correctly.

Which FOM module to use is genuinely different per directory (Zernike-standard,
Zernike+b-penalty "newfom", or without_zernike's direct-pixel parameterization) and
must be given explicitly via --fom-module (passed through to lr_scan.py too).

For each j in --j-list, writes <param-dir>/j_<j>.txt from --base-param with
j overridden, then calls lr_scan.py with that param file and a scan
directory named per --scan-dir-template (a format string with a {j}
placeholder, e.g. "lr_scan_int_para2axi_j{j}_f20").

Usage:
    python lr_scan_j.py --main-script <script> --base-param <file> \\
        --param-dir <dir> --scan-dir-template "lr_scan_int_para2axi_j{j}_f20" \\
        --fom-module {zernike,zernike_newfom,local} \\
        [--j-list 4,5,6,7,8,9,10] \\
        [--lr-min X] [--lr-max Y] [--n-lr N] [--lr-list "a,b,c"] [--scan-steps S]

Any of --lr-min/--lr-max/--n-lr/--lr-list/--scan-steps that are given are
passed straight through to lr_scan.py; omitted ones fall back to its own
defaults. (--n-lr is lr_scan.py's own "number of lr points to geomspace"
option -- unrelated to the Zernike mode count j swept here.)
"""

import os
import sys
import argparse
import subprocess

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--main-script", required=True)
parser.add_argument("--base-param", required=True)
parser.add_argument("--param-dir", required=True)
parser.add_argument("--scan-dir-template", required=True,
                     help='Format string with a {j} placeholder, e.g. "lr_scan_int_para2axi_j{j}_f20"')
parser.add_argument("--fom-module", choices=["zernike", "zernike_newfom", "local"], default="zernike")
parser.add_argument("--j-list", default="4,5,6,7,8,9,10",
                     help="Comma-separated j values to scan. Default: 4,5,6,7,8,9,10")
parser.add_argument("--lr-min", type=float, default=None)
parser.add_argument("--lr-max", type=float, default=None)
parser.add_argument("--n-lr", type=int, default=None)
parser.add_argument("--lr-list", type=str, default=None)
parser.add_argument("--scan-steps", type=int, default=None)
args = parser.parse_args()

if args.fom_module == "local":
    # without_zernike's own real (non-shared) adjoint_intensity.py.
    sys.path.insert(0, os.getcwd())
    import adjoint_intensity as adj
elif args.fom_module == "zernike_newfom":
    import adjoint_intensity_newfom as adj
else:
    import adjoint_intensity_zernike as adj

J_LIST = [int(x) for x in args.j_list.split(",")]

os.makedirs(args.param_dir, exist_ok=True)

base_params = adj.initialize_variables_from_file(args.base_param)
if base_params is None:
    raise SystemExit(f"Could not load {args.base_param}")


def write_param_file(fpath, params):
    with open(fpath, "w") as f:
        for name, value in params.items():
            f.write(f"{name} = {repr(value)}\n")


lr_scan_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lr_scan.py")

for j in J_LIST:
    params = dict(base_params)
    params["j"] = j

    fpath = os.path.join(args.param_dir, f"j_{j}.txt")
    write_param_file(fpath, params)

    scan_dir = args.scan_dir_template.format(j=j)
    print(f"\n{'='*60}\nj = {j}  ->  {fpath}  scan dir: {scan_dir}\n{'='*60}")

    cmd = ["python", lr_scan_path, args.main_script, fpath, scan_dir, "--fom-module", args.fom_module]
    if args.lr_min is not None:
        cmd += ["--lr-min", str(args.lr_min)]
    if args.lr_max is not None:
        cmd += ["--lr-max", str(args.lr_max)]
    if args.n_lr is not None:
        cmd += ["--n-lr", str(args.n_lr)]
    if args.lr_list is not None:
        cmd += ["--lr-list", args.lr_list]
    if args.scan_steps is not None:
        cmd += ["--scan-steps", str(args.scan_steps)]
    subprocess.run(cmd, check=True)

print("\nAll j-value lr scans complete.")
