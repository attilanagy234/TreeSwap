utils_path=$(grep 'utils_path:' config.yaml | awk '{ print $2 }')
pair_n_lines=$(grep 'pair_n_lines:' config.yaml | awk '{ print $2 }')
pair_output_path=$(grep 'pair_output_path:' config.yaml | awk '{ print $2 }')
valid_src=$(grep -A2 'valid:' config.yaml | grep 'path_src:' | awk '{ print $2 }')

output_folder=run

python $utils_path/pair.py \
--original_path $valid_src \
--predicted_path $output_folder/pred.txt \
--n_lines $pair_n_lines \
--output_path $pair_output_path