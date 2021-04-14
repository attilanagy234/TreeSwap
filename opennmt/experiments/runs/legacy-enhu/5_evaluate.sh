sl=en
tl=hu

DATA_PREFIX=hunglish2
DATA_PATH=../../../../data/ftp.mokk.bme.hu/Hunglish2/combined-32-simple
ONMT_PATH=../../../OpenNMT-py

spm_model_prefix=$DATA_PATH/spm_$tl$sl

spm_decode --model=$spm_model_prefix.model < pred.txt > pred.txt.detok
sacrebleu --short $DATA_PATH/$DATA_PREFIX-valid.$tl < pred.txt.detok > final_result.txt