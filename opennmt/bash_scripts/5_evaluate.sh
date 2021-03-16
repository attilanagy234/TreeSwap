valid_tgt=$(grep -A2 'valid:' config.yaml | grep 'path_tgt:' | awk '{ print $2 }')
sp_models_path=$(grep 'sp_models_path:' config.yaml | awk '{ print $2 }')
utils_path=$(grep 'utils_path:' config.yaml | awk '{ print $2 }')
tgt_model_name=$(grep 'tgt_subword_model:' config.yaml | awk '{ print $2}' | sed 's_.*\/\(.*\)\.model_\1_')

python $utils_path/spm_decode.py --model $sp_models_path/$tgt_model_name.model -d run/pred.txt && \
sacrebleu --short $valid_tgt < run/pred.txt.sp > final_result.txt
