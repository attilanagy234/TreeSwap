#!/bin/bash

utils_path=$(yq -r .utils_path config.yaml)
pair_n_lines=$(yq -r .pair_n_lines config.yaml)
pair_output_path=$(yq -r .pair_output_path config.yaml)
test_src=$(yq -r .test_src config.yaml)
valid_src=$(yq -r .data.valid.path_src config.yaml)

echo "--Pairing sentences--"

output_folder=run

if [ "$test_src" == "null" ]; then
    file_to_evaluate_against=$valid_src
else
    file_to_evaluate_against=$test_src
fi

python $utils_path/pair.py \
--original_path $file_to_evaluate_against \
--predicted_path $output_folder/pred.txt \
--n_lines $pair_n_lines \
--output_path $pair_output_path