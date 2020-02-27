#!/bin/bash

if [ $# -eq 1 ]; then
  gen=`readlink -f $1`
else
  echo "Usage: tree_acc hypothesis"
fi

cd `dirname $0`/..

tmp=scripts/tmp
tsv=$tmp/tsv
id=$tmp/id
src=$tmp/src
tgt=$tmp/tgt
hyp=$tmp/hyp

mkdir -p $tmp

tree_acc () {
  python compute_tree_acc.py -tsv $1
}

del=`grep H- $gen | awk -F '\t' '$2=="-inf" {print $1}' | cut -d '-' -f 2 | awk '{print $1+1}'`

#grep S- $gen | awk -F '\t' '{print $1}' | sed 's/^S-//' > $id
#grep S- $gen | awk -F '\t' '{print $2}' > $src
#grep T- $gen | awk -F '\t' '{print $2}' > $tgt
#grep H- $gen | awk -F '\t' '{print $3}' > $hyp

awk 'NR==FNR{l[$0];next;} !(FNR in l)' <(echo "$del") \
  <(grep S- $gen | sort -n -k 2 -t -) | \
  awk -F '\t' '{print $1}' | sed 's/^S-//' > $id

awk 'NR==FNR{l[$0];next;} !(FNR in l)' <(echo "$del") \
  <(grep S- $gen | sort -n -k 2 -t -) | \
  awk -F '\t' '{print $2}' > $src

awk 'NR==FNR{l[$0];next;} !(FNR in l)' <(echo "$del") \
  <(grep T- $gen | sort -n -k 2 -t -) | \
  awk -F '\t' '{print $2}' > $tgt

awk 'NR==FNR{l[$0];next;} !(FNR in l)' <(echo "$del") \
  <(grep H- $gen | sort -n -k 2 -t -) | \
  awk -F '\t' '{print $3}' > $hyp

paste $id $src $hyp $tgt > $tsv

tree_acc $tsv

rm -rf $tmp
