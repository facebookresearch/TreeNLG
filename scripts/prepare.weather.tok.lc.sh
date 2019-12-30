#!/bin/bash

cd `dirname $0`/..

if [ ! -d mosesdecoder ]; then
  echo 'Cloning Moses github repository (for tokenization scripts)...'
  git clone https://github.com/moses-smt/mosesdecoder.git
fi

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl

src=mr
tgt=ar
prep=data-prep/weather
tmp=$prep/tmp
orig=data/weather

mkdir -p $orig $tmp $prep

echo -e "show data sample..."
awk -F '\t' 'NR==1 {print $1,$2}' $orig/train.tsv ; echo ""
awk -F '\t' 'NR==1 {print $3}' $orig/train.tsv ; echo ""
awk -F '\t' 'NR==1 {print $4}' $orig/train.tsv ; echo ""
awk -F '\t' 'NR==1 {print $4}' $orig/train.tsv | \
  sed 's/\[\w\+//g' | sed 's/\]//g' | awk '{$1=$1;print}' ; echo ""

echo -e "\[\w+\n_\w+" > $tmp/protected_patterns
tokenize () {
  perl $TOKENIZER -protected $tmp/protected_patterns -threads 8 -q -l en | \
  sed 's/&#91;/[/g;s/&#93;/]/g' | \
  perl $LC
}

echo "creating train..."
awk -F '\t' '{print $3}' $orig/train.tsv | tokenize > $prep/train.$src
awk -F '\t' '{print $4}' $orig/train.tsv | tokenize > $prep/train.$tgt
echo "creating valid..."
awk -F '\t' '{print $3}' $orig/val.tsv   | tokenize > $prep/valid.$src
awk -F '\t' '{print $4}' $orig/val.tsv   | tokenize > $prep/valid.$tgt
echo "creating test..."
awk -F '\t' '{print $3}' $orig/test.tsv  | tokenize > $prep/test.$src
awk -F '\t' '{print $4}' $orig/test.tsv  | tokenize > $prep/test.$tgt
