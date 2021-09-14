valid_src=$(grep -A2 'valid:' config.yaml | grep 'path_src:' | awk '{ print $2 }')
translate_model=$(grep 'translate_model:' config.yaml | awk '{ print $2 }')
src_subword_model=$(grep 'src_subword_model:' config.yaml | awk '{ print $2 }')

if [ ! -f $valid_src.sp ]; then
    srun --exclusive -p gpu --gres=mps spm_encode --model=$src_subword_model < $valid_src > $valid_src.sp
fi
srun --exclusive -p gpu --gres=mps onmt_translate -model $translate_model -src $valid_src.sp -output run/pred.txt.sp -gpu 0