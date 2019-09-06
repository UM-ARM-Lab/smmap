#!/usr/bin/env python

import argparse
import re
import pathlib

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=pathlib.Path)

    args = parser.parse_args()

    pattern = re.compile(".*examples_(\d+)_to_(\d+).npz")

    for npz_file in args.dir.glob("*.npz"):
        fname = str(npz_file.name)
        match = re.fullmatch(pattern, fname)
        complete = True

        if not match:
            complete = False
            print("npz file {} doesn't match!".format(fname))

        start = int(match.group(1))
        end = int(match.group(2))

        data_dict = np.load(npz_file)

        for example_idx in range(start, end):
            if "{}/local_env".format(example_idx) not in data_dict:
                complete = False
                print("local_env {} not in file {}".format(example_idx, fname))
            if "{}/band_pre".format(example_idx) not in data_dict:
                complete = False
                print("band_pre {} not in file {}".format(example_idx, fname))
            if "{}/band_post".format(example_idx) not in data_dict:
                complete = False
                print("band_post {} not in file {}".format(example_idx, fname))

        if complete:
            print("file {} is complete!".format(fname))


if __name__ == '__main__':
    main()
