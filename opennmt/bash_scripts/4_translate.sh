valid_src=$(grep -A2 'valid:' config.yaml | grep 'path_src:' | awk '{ print $2 }')
translate_model=$(grep -A2 'translate_model:' config.yaml | awk '{ print $2 }')

srun -p gpu --gres=mps onmt_translate -model $translate_model -src $valid_src -output run/pred.txt -gpu 0