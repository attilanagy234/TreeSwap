DATA_PREFIX=hunglish2
DATA_PATH=../../../../data/ftp.mokk.bme.hu/Hunglish2/combined-32-simple
ONMT_PATH=../../../OpenNMT-py
PROCESSED_PATH=$DATA_PATH/processed-enhu

mkdir -p run
srun --exclusive -p gpu --gres=mps python $ONMT_PATH/train.py \
-world_size 1 \
-gpu_ranks 0 \
-rnn_size 512 \
-word_vec_size 512 \
-batch_type tokens \
-batch_size 4096 \
-accum_count 4 \
-train_steps 70000 \
-max_generator_batches 32 \
-normalization tokens \
-dropout 0.1 \
-max_grad_norm 0 \
-optim sparseadam \
-encoder_type transformer \
-decoder_type transformer \
-position_encoding \
-param_init 0 \
-warmup_steps 8000 \
-learning_rate 2 \
-decay_method noam \
-label_smoothing 0.1 \
-adam_beta2 0.998 \
-data $PROCESSED_PATH/$DATA_PREFIX \
-param_init_glorot -layers 6 \
-transformer_ff 2048 \
-heads 8 \
-log_file run/output.logs