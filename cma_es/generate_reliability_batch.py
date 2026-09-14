#!/usr/bin/env python
"""
Generates N single-restart parameter files (one per independent CMA-ES run,
each with a distinct seed via seed_offset) for a basin-reliability study --
i.e. estimating what fraction of random seeds converge to the true target
basin vs. a wrong local minimum. Same physics as
para2axi_restarts_test/runs/..._restarts's own param file.

Each file has restarts=1 and a unique seed_offset, so N separate
`python main_int_para2axi_cma.py <file>` invocations (run however many at a
time you want -- see run_reliability_batch.sh) each do exactly one
independent CMA-ES run with its own seed, writing to its own output
directory (no collisions).

Usage:
    python generate_reliability_batch.py [n] [out_dir] [b]

n defaults to 100, out_dir defaults to "reliability_study_n100", b
defaults to 1.0 (the current cost function's |1-mu| penalty weight -- pass
0 to reproduce the pre-fix "old fom" cost, plain FOM/Itsq with no penalty,
for a basin-reliability comparison against the fixed cost function).
"""

import os
import sys

n = int(sys.argv[1]) if len(sys.argv) >= 2 else 100
out_dir = sys.argv[2] if len(sys.argv) >= 3 else "reliability_study_n100"
b = float(sys.argv[3]) if len(sys.argv) >= 4 else 1.0
os.makedirs(out_dir, exist_ok=True)

# Same physics/CMA-ES settings as para2axi_restarts_test's own run.
BASE_PARAMS = {
    "f_targ": 0.5,
    "f_init": 0.51,
    "j": 4,
    "nr": 2001,
    "r_maxf": 1.3,
    "nzf": 1001,
    "dz": 0.05,
    "L": 0.05,
    "steps": 200,
    "skip": 5,
    "f_num": 20,
    "ell": 0,
    "wl": 1e-06,
    "conic": -1,
    "delta": 1,
    "sg": 0,
    "order": 0,
    "stop": 1000001,
    "sigma0": 0.3,
    "b": b,
    "restarts": 1,
}

for i in range(n):
    params = dict(BASE_PARAMS)
    params["seed_offset"] = i
    file_path = os.path.join(out_dir, f"d_reliability_run{i:03d}.txt")
    with open(file_path, "w") as f:
        for name, value in params.items():
            f.write(f"{name} = {value!r}\n")

print(f"Wrote {n} param files to {out_dir}/")
