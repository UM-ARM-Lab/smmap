#!/usr/bin/env python

import argparse
import re
import pathlib

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=pathlib.Path)
    parser.add_argument("--max-files", type=int)

    args = parser.parse_args()

    pattern = re.compile(".*examples_(\d+)_to_(\d+).npz")

    num_positive = 0
    num_negative = 1
    for i, npz_file in enumerate(args.dir.glob("*.npz")):
        if args.max_files and i >= args.max_files:
            break

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
            label = data_dict["{}/label".format(example_idx)]
            if label == 1.0:
                num_positive += 1
            if label == 0.0:
                num_negative += 1

    print(num_positive, num_negative)


if __name__ == '__main__':
    main()
