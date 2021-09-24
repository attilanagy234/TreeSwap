# hu-nmt



## Building the data augmentation package

The data augmentator uses [Poetry](https://python-poetry.org/) for packaging and dependency management.

> **_NOTE:_**  Mac users need to install graphviz before following the installation.
> ```bash
> sudo chown -R $(whoami) /usr/local/bin
> brew install graphviz
> sudo chown -R root /usr/local/bin
> ```

To install all the necessary dependencies, just run:
```bash
cd src/hu_nmt
poetry install
```

All installed dependencies are written to a **poetry.lock** file.


If you already have a Poetry environment and want to resume work:
```bash
git pull
poetry update
```
Poetry update will update the lock file.

You can also launch a shell in your terminal:
```bash
poetry shell
```


To set up PyCharm with this virtual environment, just configure it as the project interpreter.

You can obtain the path for the virtualenv by:
```bash
poetry env info --path
```

## Installing Stanza models:
```python
import stanza
stanza.download('en')
```

## Installing Hungarian Spacy models
```bash
pip install https://github.com/oroszgy/spacy-hungarian-models/releases/download/hu_core_ud_lg-0.3.1/hu_core_ud_lg-0.3.1-py3-none-any.whl  
```

## Dependency parsing for Hungarian
We use [Spacy](https://github.com/oroszgy/spacy-hungarian-models) for creating the dependency parse trees for Hungarian sentences.
To precompute dependency graphs and serialize them to TSVs:
```bash
cd hu_nmt/src
./precompute_hungarian_dependency_trees.sh <data_input_path> <output_path> <file_batch_size>

```

## Dependency parsing for English
We use Stanza for English dependency parsing.
To precompute dependency graphs and serialize them to TSVs:
```bash
cd hu_nmt/src
./precompute_english_dependency_trees.sh <data_input_path> <output_path> <file_batch_size>
```

# Training models

## Setup
Create a new conda environment:
```shell
conda create --name my-env python=3.8.5
conda activate my-env
```

Install the required packages:
```shell
pip install -r requirements.txt
conda install -c conda-forge sentencepiece=0.1.95
conda install -c conda-forge sacrebleu=1.5.1
```

If you get the following error during vocabulary building:
```
ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found (...)
```

run the following lines one by one in the given order:
```shell
conda install libgcc #1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/lib/ #2
```


## Run
To train a model you need to specify a config file like [this one](hhttps://github.com/attilanagy234/hu-nmt/blob/main/opennmt/experiments/runs/huen/config.yaml) where you specify all the model parameters and data paths based on the OpenNMT documentation ([build vocab](https://opennmt.net/OpenNMT-py/options/build_vocab.html), [train](https://opennmt.net/OpenNMT-py/options/train.html), [translate](https://opennmt.net/OpenNMT-py/options/translate.html)), and also specify additional parameters for our [scripts](https://github.com/attilanagy234/hu-nmt/tree/main/opennmt/bash_scripts).

After you have set up your config.yaml file you should build your vocabularies (you only have to do this once). After the vocabs have been created you can call the [full_train.sh](https://github.com/attilanagy234/hu-nmt/blob/main/opennmt/bash_scripts/full_train.sh) script which will train your model based on your config, translate your validation set and evaluate BLEU. It will also track your execution based on the next section.

```bash
# create directory for new experiment
cd opennmt/experiments/runs
mkdir new_experiment
cd new_experiment

# create config file
vim config.yaml

# build vocabulary
../../../bash_scripts/1_build_vocab.sh

# run training with evaluation and experiment tracking
../../../bash_scripts/full_train.sh
```

You can also run the model training and evaluation steps separately with the scripts found in the `opennmt/bash_scripts` directory.

## Experiment tracking
When you run a full training or just the `8_save_history.sh` script your experiment will be tracked.

It saves the following files to the history directory in folder specified by the datetime you have ran your experiment:
- config file
- final result
- final translation of the validation set
- tensorboard logs
- translation pairs
- best model

It saves the following in the `history.tsv` file in the history directory:
- all the parameters specified in the config file (if there are nested fields they are represented as `a.b`)
- `date` - when the experiment was ran
- `history_path` - corresponding history directory
- `bleu_score` - overall BLEU score
- `bleu_score_n` - ngram BLEU score
- `git_hash` - hash of the git commit that was used

If there is a new parameter added to the config the previous runs will have `None` as a value for that parameter.
