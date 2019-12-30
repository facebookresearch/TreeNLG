#!/bin/bash

cd `dirname $0`/..

src=mr
tgt=ar
prep=data-prep/e2e
orig=data/E2E_compositional

mkdir -p $orig $prep

echo -e "show data sample..."
awk -F '\t' 'NR==1 {print $1}' $orig/train.tsv ; echo ""
awk -F '\t' 'NR==1 {print $2}' $orig/train.tsv ; echo ""
awk -F '\t' 'NR==1 {print $3}' $orig/train.tsv ; echo ""
awk -F '\t' 'NR==1 {print $3}' $orig/train.tsv | \
  sed 's/\[\w\+//g' | sed 's/\]//g' | awk '{$1=$1;print}' ; echo ""

echo "creating train..."
awk -F '\t' '{print $2}' $orig/train.tsv > $prep/train.$src
awk -F '\t' '{print $3}' $orig/train.tsv > $prep/train.$tgt
echo "creating valid..."
awk -F '\t' '{print $2}' $orig/val.tsv   > $prep/valid.$src
awk -F '\t' '{print $3}' $orig/val.tsv   > $prep/valid.$tgt
echo "creating test..."
awk -F '\t' '{print $2}' $orig/test.tsv  > $prep/test.$src
awk -F '\t' '{print $3}' $orig/test.tsv  > $prep/test.$tgt
echo "creating disc_test..."
awk -F '\t' '{print $2}' $orig/disc_test.tsv  > $prep/disc_test.$src
awk -F '\t' '{print $3}' $orig/disc_test.tsv  > $prep/disc_test.$tgt
