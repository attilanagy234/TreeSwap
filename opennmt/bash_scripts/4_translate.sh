#!/bin/bash

valid_src=$(yq -r .data.valid.path_src config.yaml)
test_src=$(yq -r .test_src config.yaml)
test_encoded_src=$(yq -r .test_encoded_src config.yaml)
translate_model=$(yq -r .translate_model config.yaml)
src_subword_model=$(yq -r .src_subword_model config.yaml)

echo "--Starting translation--"

# encode
if [ "$test_src" == "null" ]; then
    if [ "$test_encoded_src" == "null" ]; then
        if [ ! -f "$valid_src".sp ]; then
            echo "Encoding the validation set for translation"
            spm_encode --model=$src_subword_model < $valid_src > $valid_src.sp
        fi
        translation_src=$valid_src.sp
    elif [ "$test_encoded_src" != "null" ]; then
        if [ ! -f "$test_encoded_src" ]; then
            echo "Encoding the validation set to the specified $test_encoded_src location for translation"
            spm_encode --model=$src_subword_model < $valid_src > $test_encoded_src
        fi
        translation_src=$test_encoded_src
    fi
else
    if [ "$test_encoded_src" == "null" ]; then
        if [ ! -f "$test_src".sp ]; then
            echo "Encoding $test_src to $test_src.sp for translation"
            spm_encode --model=$src_subword_model < $test_src > $test_src.sp
        fi
        translation_src=$test_src.sp
    else
        if [ ! -f "$test_encoded_src" ]; then
            echo "Encoding $test_src to the specified $test_encoded_src location for translation"
            spm_encode --model=$src_subword_model < $test_src > $test_encoded_src
        fi
        translation_src=$test_encoded_src
    fi
fi

if [ -z $translation_src ]; then
    echo "translation_src variable was not set for translation. Exiting..."
    exit 1
fi

# translate
echo "Going to translate: $translation_src"
echo "Translating..."
onmt_translate -model $translate_model -src $translation_src -output run/pred.txt.sp -gpu 0