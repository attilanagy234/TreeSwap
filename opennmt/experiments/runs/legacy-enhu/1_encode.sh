sl=hu
tl=en
vocab_size=32000

DATA_PREFIX=hunglish2
DATA_PATH=../../../../data/ftp.mokk.bme.hu/Hunglish2/combined-32-simple

tmp_train_file=tmp_train.txt
spm_model_prefix=$DATA_PATH/spm_$sl$tl

# Concatenate training data
rm -rf $tmp_train_file
for f in $DATA_PATH/$DATA_PREFIX-train.$sl $DATA_PATH/$DATA_PREFIX-train.$tl
do
    cat $f >> $tmp_train_file
done

# Run sentencepiece training on the concatenated training data
spm_train \
--input=$tmp_train_file \
--model_prefix=$spm_model_prefix \
--vocab_size=$vocab_size \
--character_coverage=1 \
--input_sentence_size=500000

rm -rf $tmp_train_file
echo "Created sentencepiece model"

# Encode datasets
for data_set in train valid test
do
    for f in $DATA_PATH/$DATA_PREFIX-$data_set.$sl $DATA_PATH/$DATA_PREFIX-$data_set.$tl
    do
        spm_encode --model=$spm_model_prefix.model < $f > $f.sp
        echo "Encoded $f"
    done
done

echo "All encoded"

# Might not have run in the previous round
# onmt-build-vocab --from_format sentencepiece --from_vocab $spm_model_prefix.vocab --save_vocab $DATA_PATH/$sl$tl.vocab