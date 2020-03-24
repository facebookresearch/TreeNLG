#!/bin/bash

if [ $# -eq 2 ]; then
  gen=$(readlink -f $1)
  ref_tree=$(readlink -f $2)
else
  echo "Usage: measure_scores hypothesis reference"
  exit 0
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
tmp=scripts/tmp
hyp=$tmp/hyp
ref=$tmp/ref

mkdir -p $tmp

rmtreeinfo () {
  sed 's/\[\w\+//g' | sed 's/\]//g' | awk '{$1=$1;print}'
}


cat $ref_tree | rmtreeinfo > $ref
grep H- $gen | sort -n -k 2 -t - | awk -F '\t' '{print $3}' | rmtreeinfo > $hyp

python $SCORER -p $ref $hyp 2> /dev/null

rm -rf $tmp
