mkdir -p run
srun -p gpu --gres=mps onmt_train -config config.yaml