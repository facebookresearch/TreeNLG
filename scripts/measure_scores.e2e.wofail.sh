#!/bin/bash

if [ $# -eq 2 ]; then
  gen=`readlink -f $1`
  ref_tree=`readlink -f $2`
else
  echo "Usage: measure_scores hypothesis reference"
  exit 0
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
org=data/E2E_compositional/testset_w_refs.csv
tmp=scripts/tmp
src=$tmp/src
ref=$tmp/ref
hyp=$tmp/hyp

mkdir -p $tmp

rmtreeinfo () {
  sed 's/\[\w\+//g' | sed 's/\]//g' | awk '{$1=$1;print}'
}

del=`grep H- $gen | awk -F '\t' '$2=="-inf" {print $1}' | cut -d '-' -f 2 | awk '{print $1+1}'`

awk 'NR==FNR{l[$0];next;} !(FNR in l)' <(echo "$del") \
  $ref_tree | \
  rmtreeinfo > $ref

awk 'NR==FNR{l[$0];next;} !(FNR in l)' <(echo "$del") \
  <(grep H- $gen | sort -n -k 2 -t -) | \
  awk -F '\t' '{print $3}' | \
  rmtreeinfo > $hyp

awk 'NR==FNR{l[$0];next;} !(FNR in l)' <(echo "$del") \
  <(tail -n +2 $org | cut -d '"' -f 2) > $src

python scripts/_eval_e2e_helper.py
python $SCORER -p $ref $hyp 2> /dev/null

rm -rf $tmp
