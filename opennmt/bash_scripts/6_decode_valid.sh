valid_src=$(grep -A2 'valid:' config.yaml | grep 'path_src:' | awk '{ print $2 }')
utils_path=$(grep 'utils_path:' config.yaml | awk '{ print $2 }')
src_subword_model=$(grep 'src_subword_model:' config.yaml | awk '{ print $2}')

python $utils_path/spm_decode.py --model $src_subword_model -d $valid_src --decode_ids --output_path run/valid.txt
