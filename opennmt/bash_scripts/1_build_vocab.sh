hunglish_src=$(grep -A2 'corpus:' config.yaml | grep 'path_src:' | awk '{ print $2 }')
hunglish_tgt=$(grep -A2 'corpus:' config.yaml | grep 'path_tgt:' | awk '{ print $2 }')

utils_path=$(grep 'utils_path:' config.yaml | awk '{ print $2 }')

share_vocab=$(grep 'share_vocab:' config.yaml | awk '{ print $2 }')

src_vocab_size=$(grep 'src_vocab_size:' config.yaml | awk '{ print $2 }')
src_model_name=$(grep 'src_subword_model:' config.yaml | awk '{ print $2}' | sed 's_\(.*\)\.model_\1_')

if [ "$share_vocab" != "true" ];
then
    echo "Creating source and target sentencepiece model"
    tgt_vocab_size=$(grep 'tgt_vocab_size:' config.yaml | awk '{ print $2 }')
    tgt_model_name=$(grep 'tgt_subword_model:' config.yaml | awk '{ print $2}' | sed 's_\(.*\)\.model_\1_')

    # Run sentencepiece training on the source training data
    spm_train \
    --input=$hunglish_src \
    --model_prefix=$src_model_name \
    --vocab_size=$src_vocab_size \
    --character_coverage=1 \
    --input_sentence_size=500000

    # Run sentencepiece training on the target training data
    spm_train \
    --input=$hunglish_tgt \
    --model_prefix=$tgt_model_name \
    --vocab_size=$tgt_vocab_size \
    --character_coverage=1 \
    --input_sentence_size=500000
else
    echo "Creating shared sentencepiece model"
    tmp_train_file=tmp_train.txt
    
    # Concatenate training data
    rm -rf $tmp_train_file
    for f in $hunglish_src $hunglish_tgt
    do
        cat $f >> $tmp_train_file
    done

    # Run sentencepiece training on the concatenated training data
    spm_train \
    --input=$tmp_train_file \
    --model_prefix=$src_model_name \
    --vocab_size=$src_vocab_size \
    --character_coverage=1 \
    --input_sentence_size=500000

    rm -rf $tmp_train_file
    echo "Created shared sentencepiece model $src_path.model"
fi


onmt_build_vocab -config config.yaml -n_sample -1