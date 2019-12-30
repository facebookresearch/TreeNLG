#!/bin/bash

if [ $# -eq 1 ]; then
  gen=`readlink -f $1`
else
  echo "Usage: measure_scores hypothesis"
fi

cd `dirname $0`/..

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
ref=$tmp/ref
hyp=$tmp/hyp

mkdir -p $tmp

rmtreeinfo () {
  sed 's/\[\w\+//g' | sed 's/\]//g' | awk '{$1=$1;print}'
}

grep T- $gen | awk -F '\t' '{print $2}' | rmtreeinfo > $ref
grep H- $gen | awk -F '\t' '{print $3}' | rmtreeinfo > $hyp

python $SCORER -p $ref $hyp 2> /dev/null

rm -rf $tmp
