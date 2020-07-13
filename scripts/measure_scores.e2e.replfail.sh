#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

if [ $# -eq 3 ]; then
  gen=$(readlink -f $1)
  base=$(readlink -f $2)
  ref_tree=$(readlink -f $3)
else
  echo "Usage: measure_scores hypothesis reference baseline"
  exit
fi

cd $(dirname $0)/..

if [ ! -d e2e-metrics ]; then
  echo 'Cloning e2e-metrics github repository...'
  git clone https://github.com/tuetschek/e2e-metrics
  echo '-----------------------------------------------------'
  echo '| See README of e2e-metrics to install dependencies |'
  echo '-----------------------------------------------------'
  exit
fi

SCORER=e2e-metrics/measure_scores.py
org=data/E2E_compositional/testset_w_refs.csv
tmp=scripts/tmp
hyp=$tmp/hyp
ref=$tmp/ref
src=$tmp/src

mkdir -p $tmp

rmtreeinfo () {
  sed 's/\[\w\+//g' | sed 's/\]//g' | awk '{$1=$1;print}'
}

repl=$(grep H- $gen | awk -F '\t' '$2=="-inf" {print $1}' | cut -d '-' -f 2 | awk '{print $1+1}')
awk -F '\t' 'NR==FNR {l[$0];next;} !(FNR in l) {print $3} (FNR in l) {print $6}' \
  <(echo "$repl") <(paste <(grep H- $gen | sort -n -k 2 -t -) <(grep H- $base | sort -n -k 2 -t -)) | \
  rmtreeinfo > $hyp

cat $ref_tree | rmtreeinfo > $ref
tail -n +2 $org | cut -d '"' -f 2 > $src

python scripts/_eval_e2e_helper.py
python $SCORER -p $ref $hyp 2> /dev/null

rm -rf $tmp
