hunglish_src=$(grep -A2 'corpus:' config.yaml | grep 'path_src:' | awk '{ print $2 }')
hunglish_tgt=$(grep -A2 'corpus:' config.yaml | grep 'path_tgt:' | awk '{ print $2 }')

sp_models_path=$(grep 'sp_models_path:' config.yaml | awk '{ print $2 }')
utils_path=$(grep 'utils_path:' config.yaml | awk '{ print $2 }')

src_vocab_size=$(grep 'src_vocab_size:' config.yaml | awk '{ print $2 }')
tgt_vocab_size=$(grep 'tgt_vocab_size:' config.yaml | awk '{ print $2 }')
src_model_name=$(grep 'src_subword_model:' config.yaml | awk '{ print $2}' | sed 's_.*\/\(.*\)\.model_\1_')
tgt_model_name=$(grep 'tgt_subword_model:' config.yaml | awk '{ print $2}' | sed 's_.*\/\(.*\)\.model_\1_')

src_path=$sp_models_path/$src_model_name
tgt_path=$sp_models_path/$tgt_model_name

python $utils_path/spm_train.py -d $hunglish_src -p $src_model_name --vocab-size $src_vocab_size && \
mv $src_model_name.model $src_path.model && \
mv $src_model_name.vocab $src_path.vocab && \
python $utils_path/spm_train.py -d $hunglish_tgt -p $tgt_model_name --vocab-size $tgt_vocab_size && \
mv $tgt_model_name.model $tgt_path.model && \
mv $tgt_model_name.vocab $tgt_path.vocab && \
onmt_build_vocab -config config.yaml -n_sample -1