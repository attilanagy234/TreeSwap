# TreeSwap

[![Tests](https://github.com/attilanagy234/syntax-augmentation-nmt/actions/workflows/run-tests.yaml/badge.svg)](https://github.com/attilanagy234/syntax-augmentation-nmt/actions/workflows/run-tests.yaml)

Complimentary code for our paper [TreeSwap: Data Augmentation for Machine Translation via Dependency Subtree Swapping](https://aclanthology.org/2023.ranlp-1.82/) accepted at RANLP 2023.

# Building the data augmentation package

The data augmentator uses [Poetry](https://python-poetry.org/) for packaging and dependency management.

> **_NOTE:_**  Mac users need to install graphviz before following the installation.
> ```bash
> sudo chown -R $(whoami) /usr/local/bin
> brew install graphviz
> sudo chown -R root /usr/local/bin
> ```

## Current server setup

To use all the features in the repo
```bash
conda create --name my-env python=3.8.5
conda activate my-env

pip install -r requirements.txt
pip3 install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
conda install -c conda-forge sentencepiece=0.1.95 sacrebleu=1.5.1 fasttext=0.9.2 yq=2.13.0
conda install libgcc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/lib/

cd src
poetry install
```

## Setup
To install all the necessary dependencies, just run:
```bash
cd src/hu_nmt
poetry install
pip3 install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
conda install -c conda-forge fasttext=0.9.2 yq=2.13.0
```

### Download model for language detection (used in preprocessing)
```bash
wget -O /tmp/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
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

## Running augmentation
The `augment.sh` uses the following parameters from `config.yaml`:
- `data.original` 
- augmentation hyperparameters:
  - `augmentation_type`: `ged`/`edge_mapper`/`base`
  - `similarity_threshold`
  - `augmentation_ratio`

[Example augmentation config](https://github.com/attilanagy234/TreeSwap/tree/main/opennmt/experiments/runs/simple_aug_example/config.yaml)

```bash
# create directory for new experiment
cd opennmt/experiments/runs/simple_aug_example

# set the data path in the config file
vim config.yaml

../../../bash_scripts/augment.sh 
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
pip3 install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
conda install -c conda-forge sentencepiece=0.1.95 sacrebleu=1.5.1 fasttext=0.9.2 yq=2.13.0
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

## Datasets
The preprocessed datasets and the train/dev/test splits used in the experiments for our paper: `TreeSwap: Data Augmentation for Machine Translation via Dependency Subtree Swapping` can be found [here](https://drive.google.com/drive/folders/1VgBj4tJeT0SjDwIyylUF0pH5YIFpVQ4y).


## Trained models
Our trained models from the paper `Syntax-based data augmentation for Hungarian-English machine translation` for `hu-en` and `en-hu` specifically, are available on the HuggingFace Model Hub with usage steps:
- [HU-EN](https://huggingface.co/SZTAKI-HLT/opennmt-hu-en)
- [EN-HU](https://huggingface.co/SZTAKI-HLT/opennmt-en-hu)

## Citation

If you use our method please cite the following papers:

```
@inproceedings{nagy-etal-2023-treeswap,
    title = "{T}ree{S}wap: Data Augmentation for Machine Translation via Dependency Subtree Swapping",
    author = "Nagy, Attila  and
      Lakatos, Dorina  and
      Barta, Botond  and
      {\'A}cs, Judit",
    editor = "Mitkov, Ruslan  and
      Angelova, Galia",
    booktitle = "Proceedings of the 14th International Conference on Recent Advances in Natural Language Processing",
    month = sep,
    year = "2023",
    address = "Varna, Bulgaria",
    publisher = "INCOMA Ltd., Shoumen, Bulgaria",
    url = "https://aclanthology.org/2023.ranlp-1.82",
    pages = "759--768",
    abstract = "Data augmentation methods for neural machine translation are particularly useful when limited amount of training data is available, which is often the case when dealing with low-resource languages. We introduce a novel augmentation method, which generates new sentences by swapping objects and subjects across bisentences. This is performed simultaneously based on the dependency parse trees of the source and target sentences. We name this method TreeSwap. Our results show that TreeSwap achieves consistent improvements over baseline models in 4 language pairs in both directions on resource-constrained datasets. We also explore domain-specific corpora, but find that our method does not make significant improvements on law, medical and IT data. We report the scores of similar augmentation methods and find that TreeSwap performs comparably. We also analyze the generated sentences qualitatively and find that the augmentation produces a correct translation in most cases. Our code is available on Github.",
}

```

```
@inproceedings {nagy2023syntax,
    title = {{Data Augmentation for Machine Translation via Dependency Subtree Swapping}},
    author = {Nagy, Attila and Lakatos, Dorina and Barta, Botond and Nanys, Patrick and {\'{A}}cs, Judit},
    booktitle = {XIX. Conference on Hungarian Computational Linguistics.},
    year = {2023},
}
```
