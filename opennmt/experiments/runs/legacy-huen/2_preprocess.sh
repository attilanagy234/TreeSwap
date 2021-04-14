sl=hu
tl=en

DATA_PREFIX=hunglish2
DATA_PATH=../../../../data/ftp.mokk.bme.hu/Hunglish2/combined-32-simple
ONMT_PATH=../../../OpenNMT-py
PROCESSED_PATH=$DATA_PATH/processed

mkdir -p $PROCESSED_PATH

python $ONMT_PATH/preprocess.py \
-train_src $DATA_PATH/$DATA_PREFIX-train.$sl.sp \
-train_tgt $DATA_PATH/$DATA_PREFIX-train.$tl.sp \
-valid_src $DATA_PATH/$DATA_PREFIX-valid.$sl.sp \
-valid_tgt $DATA_PATH/$DATA_PREFIX-valid.$sl.sp \
-save_data $PROCESSED_PATH/$DATA_PREFIX-2 \
-src_seq_length 100 \
-tgt_seq_length 100 \
-share_vocab