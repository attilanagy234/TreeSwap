#!/bin/bash
base_dir=$(dirname "$0")

$base_dir/2_train_exclusive.sh && \
$base_dir/3_remove_unused_models.sh && \
$base_dir/4_translate.sh && \
$base_dir/5_evaluate.sh && \
# $base_dir/6_decode_valid.sh
$base_dir/7_pair.sh && \
$base_dir/8_save_history.sh