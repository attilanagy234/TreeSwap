sl=en
tl=hu

DATA_PREFIX=hunglish2
DATA_PATH=../../../../data/ftp.mokk.bme.hu/Hunglish2/combined-32-simple
ONMT_PATH=../../../OpenNMT-py

srun --exclusive -p gpu --gres=mps python $ONMT_PATH/translate.py \
-model model_step_70000.pt \
-src $DATA_PATH/$DATA_PREFIX-valid.$sl.sp \
-replace_unk \
-alpha 0.6 \
-beta 0.0 \
-beam_size 5 \
-length_penalty wu \
-coverage_penalty wu \
-tgt $DATA_PATH/$DATA_PREFIX-valid.$tl.sp \
-gpu 0