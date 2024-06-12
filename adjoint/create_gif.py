#!/usr/bin/env python
# coding: utf-8

import imageio.v2 as imageio
import os
import sys


def count_files(directory):
    try:
        # List all files in the directory
        files = os.listdir(directory)

        # Count the number of files
        num_files = len(files)

        return num_files
    except FileNotFoundError:
        print(f"The directory '{directory}' does not exist.")
        return 0


# Example usage:

if len(sys.argv) != 3:
    print("Usage: python create_gif.py <name of run directory> <fps>")
    sys.exit(1)

path = sys.argv[1]
fps = int(sys.argv[2])
steps = count_files(path + "/g/")

gr = imageio.get_writer(path + "/grad_%d.mp4" % (steps), fps=fps)
phr_f = imageio.get_writer(path + "/phi_r_f_%d.mp4" % (steps), fps=fps)
phr_oa = imageio.get_writer(path + "/phi_r_oa_%d.mp4" % (steps), fps=fps)
s = imageio.get_writer(path + "/s%d.mp4" % (steps), fps=fps)
for i in range(0, steps, 1):
    gr_f = path + "/g/g_%d.png" % i
    pr_f = path + "/phr_f/phr_f_%d.png" % i
    proa_f = path + "/pr_oa/pr_oa_%d.png" % i
    s_f = path + "/s/s_%d.png" % i

    gr_im = imageio.imread(gr_f)
    pr_im = imageio.imread(pr_f)
    proa_im = imageio.imread(proa_f)
    s_im = imageio.imread(s_f)

    gr.append_data(gr_im)
    phr_f.append_data(pr_im)
    phr_oa.append_data(proa_im)
    s.append_data(s_im)

    if i % 50 == 0:
        print(i)
gr.close()
phr_f.close()
phr_oa.close
s.close()
