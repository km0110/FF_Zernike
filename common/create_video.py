#!/usr/bin/env python
# coding: utf-8

# Shared across adam, cma_es, and without_zernike -- lives in common/.

import glob
import os
import re
import sys

import imageio.v2 as imageio

FRAME_RE = re.compile(r"I_oa_(\d+)\.png$")


if len(sys.argv) not in (3, 4):
    print("Usage: python create_video.py <name of run directory> <fps> [output directory]")
    sys.exit(1)

path = sys.argv[1]
fps = int(sys.argv[2])
out_dir = sys.argv[3] if len(sys.argv) == 4 else path
os.makedirs(out_dir, exist_ok=True)

frame_files = glob.glob(os.path.join(path, "I_OA", "I_oa_*.png"))
frame_files.sort(key=lambda f: int(FRAME_RE.search(f).group(1)))

if not frame_files:
    print(f"No I_OA frames found in {path}")
    sys.exit(1)

last_step = int(FRAME_RE.search(frame_files[-1]).group(1))

i_oa = imageio.get_writer(os.path.join(out_dir, "I_oa_%d.mp4" % last_step), fps=fps)

for idx, f in enumerate(frame_files):
    i_oa.append_data(imageio.imread(f))

    if idx % 50 == 0:
        print(idx)

i_oa.close()
