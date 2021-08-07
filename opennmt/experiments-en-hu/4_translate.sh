hunglish_path=../../../../data/ftp.mokk.bme.hu/Hunglish2/combined-en-hu

srun --exclusive -p gpu --gres=mps onmt_translate -model run/model_no_qoutes_step_340000.pt -src $hunglish_path/hunglish2-short-valid.en -output run/pred.txt -gpu 0 --beam_size 8 --n_best 8 --batch_size 64 --random_sampling_temp 10