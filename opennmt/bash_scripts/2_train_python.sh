mkdir -p run
echo "--Starting training--"
srun --exclusive -p gpu --gres=mps python ../../../../BERT/main.py --config config.yaml
