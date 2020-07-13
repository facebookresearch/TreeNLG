#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import re

from constrained_decoding.constraint_checking import TreeConstraints

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute tree accuracy')
    parser.add_argument('-tsv', type=str, help='tsv file expected in format id, input, pred, others...')
    parser.add_argument('--order-constr', action='store_true', help='activate order constraint')
    args = parser.parse_args()
    with open(args.tsv, 'r') as f:
        lines = [l.strip().split('\t')[1:3] for l in f.readlines()]
    correct = 0
    for k, (src, tgt) in enumerate(lines):
        print(f'progress: %{100*(k+1)/len(lines):.2f} ({k+1}/{len(lines)})', end='\r')
        src_tree = TreeConstraints(src.strip(), args.order_constr)
        tgt_nt = re.compile(r'(\[\S+|\])').findall(tgt.strip())
        for i, w in enumerate(tgt_nt):
            if not src_tree.next_token(w, i):
                break
        else:
            if src_tree.meets_all():
                correct += 1
    print(
        'Tree accuracy: {:.2f} ({} / {})'.format(
            correct / len(lines) * 100, correct, len(lines)
        )
    )
