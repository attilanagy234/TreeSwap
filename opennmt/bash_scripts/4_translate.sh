valid_src=$(grep '^[[:blank:]]*[^[:blank:]#;]' config.yaml | grep -A2 'valid:' | grep 'path_src:' | awk '{ print $2 }')
translate_encoding_src=$(grep '^[[:blank:]]*[^[:blank:]#;]' config.yaml | grep 'translate_encoding_src:' | awk '{ print $2 }')
translate_encoding_tgt=$(grep '^[[:blank:]]*[^[:blank:]#;]' config.yaml | grep 'translate_encoding_tgt:' | awk '{ print $2 }')
translate_model=$(grep '^[[:blank:]]*[^[:blank:]#;]' config.yaml | grep 'translate_model:' | awk '{ print $2 }')
src_subword_model=$(grep '^[[:blank:]]*[^[:blank:]#;]' config.yaml | grep 'src_subword_model:' | awk '{ print $2 }')
tgt_subword_model=$(grep '^[[:blank:]]*[^[:blank:]#;]' config.yaml | grep 'tgt_subword_model:' | awk '{ print $2 }')

echo "--Starting translation--"

# encode
# if [ -z "$translate_encoding_tgt" ]; then
#     echo "Encoding the validation set for translation"
#     if [ ! -f $"valid_src".sp ]; then
#         srun --exclusive -p gpu --gres=mps spm_encode --model=$src_subword_model < $valid_src > $valid_src.sp
#     fi

#     translation_src=$valid_src.sp
# else
#     if [ ! -f "$translate_encoding_tgt" ]; then
#         if [ -z "$translate_src" ]; then
#             echo "Encoding the validation set to the specified $translate_encoding_tgt location for translation"
#             srun --exclusive -p gpu --gres=mps spm_encode --model=$src_subword_model < $valid_src > $translate_encoding_tgt
#         else
#             echo "Encoding $translate_src to the specified $translate_encoding_tgt location for translation"
#             srun --exclusive -p gpu --gres=mps spm_encode --model=$src_subword_model < $translate_src > $translate_encoding_tgt
#         fi
#     fi

#     translation_src=$translate_encoding_tgt
# fi

# translate
#srun --exclusive -p gpu --gres=mps onmt_translate -model $translate_model -src $translation_src -output run/pred.txt.sp -gpu 0
srun --exclusive -p gpu --gres=mps onmt_translate --model $translate_model --src $valid_src --output run/pred.txt.sp --gpu 0 --n_best 1 --batch_size 64
