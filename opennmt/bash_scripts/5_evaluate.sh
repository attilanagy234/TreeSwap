#!/bin/bash

#src_subword_model=$(yq -r .src_subword_model config.yaml)
test_tgt=$(yq -r .test_tgt config.yaml)
#valid_tgt=$(yq -r .data.valid.path_tgt config.yaml)
utils_path=$(yq -r .utils_path config.yaml)

echo "--Evaluating model--"
#spm_decode -model=$src_subword_model -input_format=piece < run/pred.txt.sp > run/pred.txt
if [ "$test_tgt" == "null" ]; then
    file_to_evaluate_against=$valid_tgt
else
    file_to_evaluate_against=$test_tgt
fi

echo "Evaluating against: $file_to_evaluate_against"
sacrebleu --short $file_to_evaluate_against < run/pred.txt > run/final_result.txt
python $utils_path/compute_meteor.py --pred run/pred.txt --test $file_to_evaluate_against --result run/final_result.txt
