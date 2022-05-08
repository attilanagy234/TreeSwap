#!/bin/bash
all_non_valid_src_paths=$(for dataset in $(yq ".data | keys" config.yaml | grep -v "valid" | grep ","); do dataset=$(echo $dataset | sed "s/\",*//g"); yq -r .data.$dataset.path_src config.yaml; done)
all_non_valid_tgt_paths=$(for dataset in $(yq ".data | keys" config.yaml | grep -v "valid" | grep ","); do dataset=$(echo $dataset | sed "s/\",*//g"); yq -r .data.$dataset.path_tgt config.yaml; done)

utils_path=$(yq -r .utils_path config.yaml)

share_vocab=$(yq -r .share_vocab config.yaml)

src_vocab_size=$(yq -r .src_vocab_size config.yaml)
src_model_name=$(yq -r .src_subword_model config.yaml | sed 's_\(.*\)\.model_\1_')

subword_model_type=$(yq -r .subword_model_type config.yaml)

if [ "$share_vocab" != "true" ];
then
    echo "Creating source and target sentencepiece model"
    tgt_vocab_size=$(yq -r .tgt_vocab_size config.yaml)
    tgt_model_name=$(yq -r .tgt_subword_model config.yaml | sed 's_\(.*\)\.model_\1_')

    # Concat source training data
    tmp_src_train_file=tmp_src_train.txt
    rm -rf $tmp_src_train_file
    for f in $all_non_valid_src_paths
    do
        cat $f >> $tmp_src_train_file
    done

    # Run sentencepiece training on the source training data
    spm_train \
    --input=$tmp_src_train_file \
    --model_prefix=$src_model_name \
    --vocab_size=$src_vocab_size \
    --character_coverage=1 \
    --model_type=$subword_model_type \
    --input_sentence_size=500000 \
    --hard_vocab_limit=false

    # Concat target training data
    tmp_tgt_train_file=tmp_tgt_train.txt
    rm -rf $tmp_tgt_train_file
    for f in $all_non_valid_tgt_paths
    do
        cat $f >> $tmp_tgt_train_file
    done

    # Run sentencepiece training on the target training data
    spm_train \
    --input=$tmp_tgt_train_file \
    --model_prefix=$tgt_model_name \
    --vocab_size=$tgt_vocab_size \
    --character_coverage=1 \
    --model_type=$subword_model_type \
    --input_sentence_size=500000 \
    --hard_vocab_limit=false

    # Remove temporary files
    rm -rf $tmp_src_train_file
    rm -rf $tmp_tgt_train_file
else
    echo "Creating shared sentencepiece model"
    tmp_train_file=tmp_train.txt
    
    # Concatenate training data
    rm -rf $tmp_train_file
    for f in $all_non_valid_src_paths $all_non_valid_tgt_paths
    do
        cat $f >> $tmp_train_file
    done

    # Run sentencepiece training on the concatenated training data
    spm_train \
    --input=$tmp_train_file \
    --model_prefix=$src_model_name \
    --vocab_size=$src_vocab_size \
    --character_coverage=1 \
    --model_type=$subword_model_type \
    --input_sentence_size=500000 \
    --hard_vocab_limit=false

    rm -rf $tmp_train_file
    echo "Created shared sentencepiece model $src_path.model"
fi


onmt_build_vocab -config config.yaml -n_sample -1