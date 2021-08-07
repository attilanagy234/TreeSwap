valid_src=$(grep -A2 'valid:' config.yaml | grep 'path_src:' | awk '{ print $2 }')
translate_model=$(grep 'translate_model:' config.yaml | awk '{ print $2 }')

srun -p gpu --gres=mps onmt_translate -model $translate_model -src $valid_src.sp -output run/pred.txt.sp -gpu 0