hunglish_path=../../../../data/ftp.mokk.bme.hu/Hunglish2/combined

srun -p gpu --gres=mps onmt_translate -model run/model.pt -src $hunglish_path/hunglish2-valid.hu -output run/pred.txt -gpu 0