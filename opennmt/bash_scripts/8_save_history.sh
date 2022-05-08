#!/bin/bash

HISTORY_DIR=$(yq -r .history_path config.yaml)
TSV_PATH=$HISTORY_DIR/history.tsv

save_dir=$HISTORY_DIR/$(date '+%Y-%m-%d_%H:%M:%S')
utils_path=$(yq -r .utils_path config.yaml)
pair_output_path=$(yq -r .pair_output_path config.yaml)
translate_model=$(yq -r .translate_model config.yaml)

echo "--Saving history--"

CONFIG_PATH="config.yaml"
FINAL_RESULT_PATH="run/final_result.txt"
TO_SAVE="$CONFIG_PATH $FINAL_RESULT_PATH run/pred.txt run/tensorboard $pair_output_path $translate_model"

mkdir -p $HISTORY_DIR
mkdir $save_dir
for file in $TO_SAVE
do
    cp -r $file $save_dir/
done

python3 $utils_path/save_history_to_tsv.py --yaml_path $CONFIG_PATH --result_path $FINAL_RESULT_PATH --history_path $save_dir --tsv_path $TSV_PATH