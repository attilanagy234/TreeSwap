hunglish_path=/home1/hu-nmt/data/Hunglish2/combined-32-simple
sp_models_path=../../sp_models
utils_path=../../../utils

python $utils_path/spm_decode.py --model $sp_models_path/bpe_hu.model -d run/pred.txt && \
sacrebleu --short $hunglish_path/hunglish2-valid.hu < run/pred.txt.sp > final_result.txt
