#!/bin/bash

cd `dirname $0`/..

TMPDIR=/tmp
TEXT=data-prep/weather_challenge
DESTDIR=data-bin/weather_challenge

fairseq-preprocess --source-lang mr --target-lang ar \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir $DESTDIR \
    --workers 20
