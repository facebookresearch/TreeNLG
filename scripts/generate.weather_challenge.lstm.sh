#!/bin/bash

cd $(dirname $0)/..

TMPDIR=/tmp
data=weather_challenge
model=lstm
SAVEDIR=checkpoints/$data.$model
testpfx=test
gen=gen.constr.txt

fairseq-generate data-prep/$data \
  --user-dir . `# delete this line to decode without constraints` \
  --gen-subset $testpfx \
  --path $SAVEDIR/checkpoint_best.pt \
  --dataset-impl raw \
  --max-sentences 128 \
  --beam 5 \
  --max-len-a 2 --max-len-b 50 \
  > $SAVEDIR/$gen

bash scripts/measure_scores.sh $SAVEDIR/$gen data-prep/$data/$testpfx.mr-ar.ar
bash scripts/tree_acc.sh $SAVEDIR/$gen
bash scripts/count_failure_cases.sh $SAVEDIR/$gen

if [[ "$gen" == *".constr."* ]]; then
  base=$(readlink -f $SAVEDIR/$(echo $gen | sed 's/.constr//'))
  echo -e "\nreplacing failures from $base"
  if [ ! -f $base ]; then
    echo "$base does not exist"
    exit
  fi
  bash scripts/measure_scores.replfail.sh $SAVEDIR/$gen $base data-prep/$data/$testpfx.mr-ar.ar
  bash scripts/tree_acc.replfail.sh $SAVEDIR/$gen $base
fi
