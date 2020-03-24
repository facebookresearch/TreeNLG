#!/bin/bash

if [ $# -eq 2 ]; then
  gen=$(readlink -f $1)
  base=$(readlink -f $2)
else
  echo "Usage: tree_acc hypothesis baseline"
  exit
fi

cd $(dirname $0)/..

tmp=scripts/tmp
tsv=$tmp/tsv
id=$tmp/id
src=$tmp/src
baserepl=$tmp/baserepl
hyprepl=$tmp/hyprepl
hyp=$tmp/hyp

mkdir -p $tmp

tree_acc () {
  python compute_tree_acc.py -tsv $1
}

repl=$(grep H- $gen | awk -F '\t' '$2=="-inf" {print $1}' | cut -d '-' -f 2 | awk '{print $1+1}')

grep S- $gen | sort -n -k 2 -t - | awk -F '\t' '{print $1}' | sed 's/^S-//' > $id

grep S- $gen | sort -n -k 2 -t - | awk -F '\t' '{print $2}' > $src

awk 'NR==FNR{l[$0];next;} (FNR in l)' <(echo "$repl") \
  <(grep H- $base | sort -n -k 2 -t -) > $baserepl

awk 'NR==FNR{l[$0];next;} !(FNR in l)' <(echo "$repl") \
  <(grep H- $gen | sort -n -k 2 -t -) > $hyprepl

cat $baserepl >> $hyprepl
cat $hyprepl | sort -n -k 2 -t - | awk -F '\t' '{print $3}' > $hyp

paste $id $src $hyp > $tsv

tree_acc $tsv

rm -rf $tmp
