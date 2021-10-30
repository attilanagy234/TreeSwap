tgt_subword_model=$(grep '^[[:blank:]]*[^[:blank:]#;]' config.yaml | grep 'tgt_subword_model:' | awk '{ print $2 }')
valid_tgt=$(grep '^[[:blank:]]*[^[:blank:]#;]' config.yaml |grep -A2 'valid:' | grep 'path_tgt:' | awk '{ print $2 }')

echo "--Evaluating model--"
srun spm_decode -model=$tgt_subword_model -input_format=piece < run/pred.txt.sp > run/pred.txt
sacrebleu --short $valid_tgt < run/pred.txt > run/final_result.txt
