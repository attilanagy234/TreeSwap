valid_src=$(yq -r .data.valid.path_src config.yaml)
utils_path=$(yq -r .utils_path config.yaml)
src_subword_model=$(yq -r .src_subword_model config.yaml)

python $utils_path/spm_decode.py --model $src_subword_model -d $valid_src --decode_ids --output_path run/valid.txt
