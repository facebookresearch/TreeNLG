#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

if [ $# -eq 1 ]; then
  gen=$(readlink -f $1)
else
  echo "Usage: tree_acc hypothesis"
fi

cd $(dirname $0)/..

tmp=scripts/tmp
tsv=$tmp/tsv
id=$tmp/id
src=$tmp/src
hyp=$tmp/hyp

mkdir -p $tmp

tree_acc () {
  python compute_tree_acc.py -tsv $1
}

grep S- $gen | awk -F '\t' '{print $1}' | sed 's/^S-//' > $id
grep S- $gen | awk -F '\t' '{print $2}' > $src
grep H- $gen | awk -F '\t' '{print $3}' > $hyp

paste $id $src $hyp > $tsv

tree_acc $tsv

rm -rf $tmp
