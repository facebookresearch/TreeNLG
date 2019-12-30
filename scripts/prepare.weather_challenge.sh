#!/bin/bash

cd `dirname $0`/..

src=mr
tgt=ar
prep=data-prep/weather_challenge
orig=data/weather_challenge

mkdir -p $orig $prep

echo -e "show data sample..."
awk -F '\t' 'NR==1 {print $1,$2}' $orig/train.tsv ; echo ""
awk -F '\t' 'NR==1 {print $3}' $orig/train.tsv ; echo ""
awk -F '\t' 'NR==1 {print $4}' $orig/train.tsv ; echo ""
awk -F '\t' 'NR==1 {print $4}' $orig/train.tsv | \
  sed 's/\[\w\+//g' | sed 's/\]//g' | awk '{$1=$1;print}' ; echo ""

echo "creating train..."
awk -F '\t' '{print $3}' $orig/train.tsv > $prep/train.$src
awk -F '\t' '{print $4}' $orig/train.tsv > $prep/train.$tgt
echo "creating valid..."
awk -F '\t' '{print $3}' $orig/val.tsv   > $prep/valid.$src
awk -F '\t' '{print $4}' $orig/val.tsv   > $prep/valid.$tgt
echo "creating test..."
awk -F '\t' '{print $3}' $orig/test.tsv  > $prep/test.$src
awk -F '\t' '{print $4}' $orig/test.tsv  > $prep/test.$tgt
