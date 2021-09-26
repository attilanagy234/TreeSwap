mkdir -p run
echo "--Starting training--"
srun --exclusive -p gpu --gres=mps onmt_train -config config.yaml