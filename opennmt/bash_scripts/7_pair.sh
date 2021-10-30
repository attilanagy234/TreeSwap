utils_path=$(grep '^[[:blank:]]*[^[:blank:]#;]' config.yaml | grep 'utils_path:' | awk '{ print $2 }')
pair_n_lines=$(grep '^[[:blank:]]*[^[:blank:]#;]' config.yaml | grep 'pair_n_lines:' | awk '{ print $2 }')
pair_output_path=$(grep '^[[:blank:]]*[^[:blank:]#;]' config.yaml | grep 'pair_output_path:' | awk '{ print $2 }')
valid_src=$(grep '^[[:blank:]]*[^[:blank:]#;]' config.yaml | grep -A2 'valid:' | grep 'path_src:' | awk '{ print $2 }')

echo "--Pairing sentences--"

output_folder=run

python $utils_path/pair.py \
--original_path $valid_src \
--predicted_path $output_folder/pred.txt \
--n_lines $pair_n_lines \
--output_path $pair_output_path