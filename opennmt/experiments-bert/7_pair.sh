utils_path=../../../utils
output_folder=../../runs/bert/run

python $utils_path/pair.py \
--original_path $output_folder/valid.txt \
--predicted_path $output_folder/pred.txt.sp \
--n_lines 50 \
--output_path $output_folder/paired.txt \