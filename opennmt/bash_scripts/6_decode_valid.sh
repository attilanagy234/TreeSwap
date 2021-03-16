valid_src=$(grep -A2 'valid:' config.yaml | grep 'path_src:' | awk '{ print $2 }')
sp_models_path=$(grep 'sp_models_path:' config.yaml | awk '{ print $2 }')
utils_path=$(grep 'utils_path:' config.yaml | awk '{ print $2 }')
src_model_name=$(grep 'src_subword_model:' config.yaml | awk '{ print $2}' | sed 's_.*\/\(.*\)\.model_\1_')

python $utils_path/spm_decode.py --model $sp_models_path/$src_model_name.model -d $valid_src --decode_ids --output_path run/valid.txt
