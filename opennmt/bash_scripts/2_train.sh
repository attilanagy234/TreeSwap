mkdir -p run
echo "--Starting training--"
srun -p gpu --gres=mps onmt_train -config config.yaml