src_subword_model=$(grep 'src_subword_model:' config.yaml | awk '{ print $2 }')
valid_tgt=$(grep -A2 'valid:' config.yaml | grep 'path_tgt:' | awk '{ print $2 }')
# utils_path=$(grep 'utils_path:' config.yaml | awk '{ print $2 }')
# tgt_model_name=$(grep 'tgt_subword_model:' config.yaml | awk '{ print $2}' | sed 's_.*\/\(.*\)\.model_\1_')

# python $utils_path/spm_decode.py --model $sp_models_path/$tgt_model_name.model -d run/pred.txt && \
spm_decode -model=$src_subword_model -input_format=piece < run/pred.txt.sp > run/pred.txt
sacrebleu --short $valid_tgt < run/pred.txt > run/final_result.txt
