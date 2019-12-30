#!/bin/bash

cd `dirname $0`/..

TMPDIR=/tmp
TEXT=data-prep/e2e
DESTDIR=data-bin/e2e

fairseq-preprocess --source-lang mr --target-lang ar \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test,$TEXT/disc_test \
    --destdir $DESTDIR \
    --workers 20

cd $DESTDIR
for f in test1.*; do
  mv "$f" "${f/test1/disc_test}"
done
