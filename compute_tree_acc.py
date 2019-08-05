#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse

from tree_accuracy.tree_accuracy import compare_trees, scenario_to_tree, sequence_to_tree


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute tree accuracy")
    # TSV file expected in format id, input, pred, target
    parser.add_argument("-tsv", type=str)
    args = parser.parse_args()
    with open(args.tsv, "r") as f:
        lines = [l.strip().split("\t") for l in f.readlines()]
    print("Loaded {} lines".format(len(lines)))
    correct = 0
    for line in lines:
        scenario_tree = scenario_to_tree(line[1].split(" "))
        pred_tree = sequence_to_tree(line[2].split(" "))
        if compare_trees(scenario_tree, pred_tree):
            correct += 1
    print(
        "Tree accuracy: {:.2f} ({} / {})".format(
	        correct / len(lines) * 100, correct, len(lines)
        )
    )
