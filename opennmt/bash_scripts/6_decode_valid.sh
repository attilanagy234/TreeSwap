valid_src=$(grep '^[[:blank:]]*[^[:blank:]#;]' config.yaml | grep -A2 'valid:' | grep 'path_src:' | awk '{ print $2 }')
utils_path=$(grep '^[[:blank:]]*[^[:blank:]#;]' config.yaml | grep 'utils_path:' | awk '{ print $2 }')
src_subword_model=$(grep '^[[:blank:]]*[^[:blank:]#;]' config.yaml | grep 'src_subword_model:' | awk '{ print $2}')

python $utils_path/spm_decode.py --model $src_subword_model -d $valid_src --decode_ids --output_path run/valid.txt
