#!/bin/bash

cd `dirname $0`/..

if [ ! -d glove ]; then
  mkdir -p glove
  cd glove
  echo 'Downloading glove.840B.300d.zip...'
  wget http://nlp.stanford.edu/data/glove.840B.300d.zip
  unzip glove.840B.300d.zip
  rm glove.840B.300d.zip
fi

