#!/bin/bash
base_dir=$(dirname "$0")
log_file=$(yq -r .log_file config.yaml)

function succeded() {
    grep "0" <<< "${PIPESTATUS[0]}" &> /dev/null
}

$base_dir/2_train.sh |& tee -a $log_file; succeded && \
$base_dir/3_remove_unused_models.sh |& tee -a $log_file; succeded && \
$base_dir/4_translate.sh |& tee -a $log_file; succeded && \
$base_dir/5_evaluate.sh |& tee -a $log_file; succeded && \
$base_dir/7_pair.sh |& tee -a $log_file; succeded && \
$base_dir/8_save_history.sh |& tee -a $log_file; succeded