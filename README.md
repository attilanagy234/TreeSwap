# hu-nmt



## Building the data augmentation package

The data augmentator uses [Poetry](https://python-poetry.org/) for packaging and dependency management.

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

You also need to download the english pipeline for Spacy
```bash
 python -m spacy download en
```

To set up PyCharm with this virtual environment, just configure it as the project interpreter.

You can obtain the path for the virtualenv by:
```bash
poetry env info --path
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