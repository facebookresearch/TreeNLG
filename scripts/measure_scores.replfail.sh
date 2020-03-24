#!/bin/bash

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
tmp=scripts/tmp
baserepl=$tmp/baserepl
hyprepl=$tmp/hyprepl
hyp=$tmp/hyp
ref=$tmp/ref

mkdir -p $tmp

rmtreeinfo () {
  sed 's/\[\w\+//g' | sed 's/\]//g' | awk '{$1=$1;print}'
}

repl=$(grep H- $gen | awk -F '\t' '$2=="-inf" {print $1}' | cut -d '-' -f 2 | awk '{print $1+1}')

awk 'NR==FNR{l[$0];next;} (FNR in l)' <(echo "$repl") \
  <(grep H- $base | sort -n -k 2 -t -) > $baserepl

awk 'NR==FNR{l[$0];next;} !(FNR in l)' <(echo "$repl") \
  <(grep H- $gen | sort -n -k 2 -t -) > $hyprepl

cat $baserepl >> $hyprepl
cat $hyprepl | sort -n -k 2 -t - | awk -F '\t' '{print $3}' | rmtreeinfo > $hyp

cat $ref_tree | rmtreeinfo > $ref

python $SCORER -p $ref $hyp 2> /dev/null

rm -rf $tmp
