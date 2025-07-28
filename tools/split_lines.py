#!/usr/bin/env python3

import sys
import random
import numpy as np
from tqdm import tqdm
def main():
    if len(sys.argv) != 5:
        print(f"Usage: {sys.argv[0]} <input_file> <train_ratio> <valid_ratio> <test_ratio>")
        sys.exit(1)

    input_file = sys.argv[1]
    train_ratio = int(sys.argv[2])
    valid_ratio = int(sys.argv[3])
    test_ratio  = int(sys.argv[4])

    # -----------------------------
    # First pass: count total lines
    # -----------------------------
    line_count = 0
    print(f"Counting lines in {input_file} ...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for _ in f:
            line_count += 1

    print(f"Total lines: {line_count}")
    # -----------------------------------------------------
    # Determine how many lines should go to each split
    # -----------------------------------------------------
    total_ratio = train_ratio + valid_ratio + test_ratio
    train_count = int(line_count * train_ratio / total_ratio)
    valid_count = int(line_count * valid_ratio / total_ratio)
    # Let test_count fill in whatever is left
    test_count = line_count - train_count - valid_count

    # ---------------------------------------------
    # Randomly shuffle indices and pick subsets
    # ---------------------------------------------
    # indices = list(range(line_count))
    # random.shuffle(indices)
    indices = np.random.permutation(line_count)

    train_indices = set(indices[:train_count])
    valid_indices = set(indices[train_count:train_count + valid_count])
    test_indices  = set(indices[train_count + valid_count:])

    # ------------------------------------------
    # Second pass: write lines to the 3 files
    # ------------------------------------------
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(input_file + '.train', 'w', encoding='utf-8') as ftrain, \
         open(input_file + '.valid', 'w', encoding='utf-8') as fvalid, \
         open(input_file + '.test',  'w', encoding='utf-8') as ftest:

        for i, line in tqdm(enumerate(fin), total=line_count):
            if i in train_indices:
                ftrain.write(line)
            elif i in valid_indices:
                fvalid.write(line)
            else:
                ftest.write(line)


if __name__ == '__main__':
    main()