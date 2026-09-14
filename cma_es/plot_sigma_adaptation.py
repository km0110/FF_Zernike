#!/usr/bin/env python
"""
CMA-ES step-size (sigma) adaptation vs. overall-best cost, per generation,
for one or more completed runs overlaid on shared axes -- parsed directly
from each run's saved stdout log (main_int_para2axi_cma.py prints sigma
every generation but doesn't save it to any CSV, so the log is the only
record of it).

Narrative this supports: sigma is CMA-ES's own live signal of how settled
its search is. A steadily shrinking sigma means it's converging cleanly
onto one basin; a sigma that stalls or bounces back up signals it's still
straddling a ridge. Overlaying a target-basin run against a wrong-basin run
(especially one of the runs that stopped via 'tolfun', i.e. CMA-ES itself
believed it had converged) shows whether the algorithm was ever actually
uncertain, or whether it collapsed early and confidently onto the wrong
answer -- mechanistic evidence behind the reliability-study success rate,
not just a repeat of that number.

Usage:
    python plot_sigma_adaptation.py <out_prefix> <log_file>=<label> [<log_file>=<label> ...]

Output is <out_prefix>.pdf/.png.
"""

import re
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

font = {"size": 15}
plt.rc("font", **font)
plt.rcParams["lines.linewidth"] = 2
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

COLORS = ["tab:green", "tab:orange", "tab:red", "tab:purple", "tab:brown"]

LINE_RE = re.compile(
    r"GEN (\d+)\s+best-of-gen cost=FOM/Itsq\+(?:lam|b)\*\|1-mu\|:([-\d.eE+]+)\s+"
    r"overall best:([-\d.eE+]+)\s+sigma=([-\d.eE+]+)"
)


def parse_log(log_path):
    generations, best_costs, sigmas = [], [], []
    with open(log_path) as f:
        for line in f:
            m = LINE_RE.search(line)
            if m:
                generations.append(int(m.group(1)))
                best_costs.append(float(m.group(3)))
                sigmas.append(float(m.group(4)))
    if not generations:
        raise RuntimeError(f"no 'GEN ... sigma=...' lines found in {log_path} -- "
                            f"wrong log file, or run used a different script version?")
    return np.array(generations), np.array(best_costs), np.array(sigmas)


if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(1)

out_prefix = sys.argv[1]
runs = []
for arg in sys.argv[2:]:
    log_path, _, label = arg.partition("=")
    if not label:
        raise SystemExit(f"expected <log_file>=<label>, got: {arg}")
    generations, best_costs, sigmas = parse_log(log_path)
    runs.append({"label": label, "generations": generations, "cost": best_costs, "sigma": sigmas})
    print(f"{label}: {len(generations)} generations, "
          f"cost {best_costs[0]:.3e} -> {best_costs[-1]:.3e}, "
          f"sigma {sigmas[0]:.3e} -> {sigmas[-1]:.3e}")

fig, (ax_cost, ax_sigma) = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True)

for run, color in zip(runs, COLORS):
    ax_cost.plot(run["generations"], run["cost"], color=color, label=run["label"])
    ax_sigma.plot(run["generations"], run["sigma"], color=color, label=run["label"])

ax_cost.set_yscale("log")
ax_cost.set_ylabel("Cost")
ax_cost.legend(loc="best", fontsize=10)

ax_sigma.set_yscale("log")
ax_sigma.set_xlabel("Generation")
ax_sigma.set_ylabel(r"CMA-ES step size $\sigma$")

plt.tight_layout()
out_path = f"{out_prefix}.pdf"
plt.savefig(out_path, bbox_inches="tight")
png_path = out_path.rsplit(".", 1)[0] + ".png"
plt.savefig(png_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved to {out_path} and {png_path}")
