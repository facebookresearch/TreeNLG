#!/bin/bash

cd `dirname $0`/..

TMPDIR=/tmp
data=weather
model=lstm
SAVEDIR=checkpoints/$data.$model.glove
testpfx=test
gen=gen.constr.txt

fairseq-generate data-bin/$data \
  --user-dir . `# delete this line to decode without constraints` \
  --gen-subset $testpfx `# test prefix` \
  --path $SAVEDIR/checkpoint_best.pt \
  --max-sentences 128 \
  --beam 5 \
  --max-len-a 2 --max-len-b 200 \
  > $SAVEDIR/$gen

bash scripts/measure_scores.sh $SAVEDIR/$gen data-prep/$data/$testpfx.ar
bash scripts/tree_acc.sh $SAVEDIR/$gen
bash scripts/count_failure_cases.sh $SAVEDIR/$gen
