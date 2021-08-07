hunglish_path=../../../../data/ftp.mokk.bme.hu/Hunglish2/combined-en-hu
sp_models_path=../../sp_models
utils_path=../../../utils

python $utils_path/spm_decode.py --model $sp_models_path/bpe_en.model -d $hunglish_path/hunglish2-short-valid.en --decode_ids --output_path run/valid.txt
