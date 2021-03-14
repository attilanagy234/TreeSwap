hunglish_path=../../../../data/ftp.mokk.bme.hu/Hunglish2/combined-en-hu
sp_models_path=../../sp_models
utils_path=../../../utils

python $utils_path/spm_decode.py --model $sp_models_path/bpe_hu.model -d run/pred.txt && \
sacrebleu --short $hunglish_path/hunglish2-valid.hu < run/pred.txt.sp > final_result.txt
