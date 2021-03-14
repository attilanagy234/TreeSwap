hunglish_path=../../../../data/ftp.mokk.bme.hu/Hunglish2/combined-en-hu
sp_models_path=../../sp_models
utils_path=../../../utils

vocab_size=20000
src_model_name=bpe_hu
tgt_model_name=bpe_en

src_path=$sp_models_path/$src_model_name
tgt_path=$sp_models_path/$tgt_model_name

python $utils_path/spm_train.py -d $hunglish_path/hunglish2-train.hu -p $src_model_name --vocab-size $vocab_size && \
mv $src_model_name.model $src_path.model && \
mv $src_model_name.vocab $src_path.vocab && \
python $utils_path/spm_train.py -d $hunglish_path/hunglish2-train.en -p $tgt_model_name --vocab-size $vocab_size && \
mv $tgt_model_name.model $tgt_path.model && \
mv $tgt_model_name.vocab $tgt_path.vocab && \
onmt_build_vocab -config config.yaml -n_sample -1