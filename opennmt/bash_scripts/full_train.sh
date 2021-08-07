#!/bin/bash
BASEDIR=$(dirname "$0")

$BASEDIR/2_train_exclusive.sh
$BASEDIR/3_remove_unused_models.sh
$BASEDIR/4_translate.sh
$BASEDIR/5_evaluate.sh
$BASEDIR/6_decode_valid.sh
$BASEDIR/7_pair.sh

HISTORY_DIR=../../history
SAVE_DIR=$HISTORY_DIR/$(date '+%Y-%m-%d_%H:%M:%S')
TO_SAVE="config.yaml run/final_result.txt run/pred.txt.sp run/pair_out.txt run/tensorboard"

mkdir -p $HISTORY_DIR
mkdir $SAVE_DIR
for FILE in $TO_SAVE
do
    cp -r $FILE $SAVE_DIR/
done