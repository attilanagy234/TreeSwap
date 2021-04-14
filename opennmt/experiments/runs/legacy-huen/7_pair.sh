sl=hu
tl=en

DATA_PREFIX=hunglish2
DATA_PATH=../../../../data/ftp.mokk.bme.hu/Hunglish2/combined-32-simple

utils_path=../../../utils
pair_n_lines=20
pair_output_path=pairs.txt

python $utils_path/pair.py \
--original_path $DATA_PATH/$DATA_PREFIX-valid.$sl \
--predicted_path pred.txt.detok \
--n_lines $pair_n_lines \
--output_path $pair_output_path