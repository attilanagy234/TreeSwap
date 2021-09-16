mkdir -p run
srun --exclusive -p gpu --gres=mps onmt_train -config config.yaml