src_subword_model=$(grep 'src_subword_model:' config.yaml | awk '{ print $2 }')
valid_tgt=$(grep -A2 'valid:' config.yaml | grep 'path_tgt:' | awk '{ print $2 }')

srun --exclusive -p gpu --gres=mps spm_decode -model=$src_subword_model -input_format=piece < run/pred.txt.sp > run/pred.txt
sacrebleu --short $valid_tgt < run/pred.txt > run/final_result.txt
