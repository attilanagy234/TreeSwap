hunglish_src=$(grep '^[[:blank:]]*[^[:blank:]#;]' config.yaml | grep -A2 'corpus:' | grep 'path_src:' | awk '{ print $2 }')
hunglish_tgt=$(grep '^[[:blank:]]*[^[:blank:]#;]' config.yaml | grep -A2 'corpus:' | grep 'path_tgt:' | awk '{ print $2 }')

utils_path=$(grep '^[[:blank:]]*[^[:blank:]#;]' config.yaml | grep 'utils_path:' | awk '{ print $2 }')

subword_model_type=$(grep '^[[:blank:]]*[^[:blank:]#;]' config.yaml | grep 'subword_model_type:' | awk '{ print $2 }')
src_vocab_size=$(grep '^[[:blank:]]*[^[:blank:]#;]' config.yaml | grep 'src_vocab_size:' | awk '{ print $2 }')
tgt_vocab_size=$(grep '^[[:blank:]]*[^[:blank:]#;]' config.yaml | grep 'tgt_vocab_size:' | awk '{ print $2 }')

src_model_name=$(grep '^[[:blank:]]*[^[:blank:]#;]' config.yaml | grep 'src_subword_model:' | awk '{ print $2}' | sed 's_\(.*\)\.model_\1_')
tgt_model_name=$(grep '^[[:blank:]]*[^[:blank:]#;]' config.yaml | grep 'tgt_subword_model:' | awk '{ print $2}' | sed 's_\(.*\)\.model_\1_')

mkdir $(echo $src_model_name | cut -d '/' -f1)
mkdir $(echo $tgt_model_name | cut -d '/' -f1)

echo "Creating source sentencepiece model"
srun python $utils_path/spm_train.py -d $hunglish_src -p $src_model_name --vocab-size $src_vocab_size --character-coverage=1 --model-type=$subword_model_type

#echo "Creating target sentencepiece model"
srun python $utils_path/spm_train.py -d $hunglish_tgt -p $tgt_model_name --vocab-size $tgt_vocab_size --character-coverage=1 --model-type=$subword_model_type

#mv $src_model_name.model $src_path.model && \
# mv $src_model_name.vocab $src_path.vocab && \
# mv $tgt_model_name.model $tgt_path.model && \
# mv $tgt_model_name.vocab $tgt_path.vocab

srun onmt_build_vocab -config config.yaml -n_sample -1
