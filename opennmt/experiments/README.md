# How to run experiments

This readme tends to serve as a handy guide on how to run experiments quickly using the scripts we wrote. The bash
scripts are used to wrap two things:
- the Python [entrypoints](https://github.com/attilanagy234/syntax-augmentation-nmt/tree/main/src/hu_nmt/data_augmentator/entrypoints)
in the augmentator library (preprocessing, dependency parsing, augmentation)
- openNMT (for training NMT models)

There might be cases, where it's just simpler to run the entrypoint scripts on their own (e.g. you want to preprocess
a dataset quickly, just call the preprocessing entrypoint). If you do so, you can always put the resulting data file in
the correct path (specified in the shared config) and keep using the scripts below.


## 0. One config to rule them all

We use a [shared, hierarchical yaml configuration](https://github.com/attilanagy234/syntax-augmentation-nmt/blob/main/opennmt/experiments/runs/simple_train_example/config.yaml)
to define the parameters of every script.

In case the one gigantic hierarchical config seems like it is too much, you can break it up into smaller configs
depending on which scripts you are using.

### Warning ⚠️
There might be some unused parameters in the config that are not relevant for the below scripts, please disregard these.
(eg. multi_train)


## 1. Preprocessing, Dependecy parsing and Augmentation

All of the above three steps can be run using a single script: [augment.sh](../bash_scripts/augment.sh). 

The script contains three main functions `preprocess_data_for_augmentation`, `create_dependency_trees` and `augment`.
These scripts are executed in the above order, in case the data is not already available: before each step, 
the script checks if the output of the current step is already there, in this case the step is skipped
(to avoid unnecessary computations).

### Warning ⚠️
Partly failed runs are not handled properly, so e.g. if your dependency parsing fails half way through and some files
are already there, it's possible that in your next run the dependency parsing step will be skipped. For this reason,
always make sure to check what files are there in your base folder and remove any unnecessary or incomplete files before
starting a new run. 

### Sections used for configuration
- Augmentation
- Data and vocab config

The second is needed, because the script will take the input data from
[this path](https://github.com/attilanagy234/syntax-augmentation-nmt/blob/main/opennmt/experiments/runs/simple_train_example/config.yaml#L113) 
(both src and trg)

## 2. Build vocabulary

Building a vocabulary is done by the OpenNMT library (more guide
[here](https://opennmt.net/OpenNMT-py/examples/Translation.html#step-1-build-the-vocabulary)).
The [1_build_vocab.sh](../bash_scripts/1_build_vocab.sh) script can be used to perform the steps based on a config.

The script concatenates all the non-valid datasets from the config and builds a vocabulary based on them.
There are two modes that one could use:
1. **use a shared vocabulary**
   1. concatenate the source and target training data and create one sentencepiece model for both.
2. **use separate vocabularies**
   1. concatenate all the source and target training data separately, creating two separate sentencepiece models
   and vocabularies

### Sections used for configuration
- Script configs
- Data and vocab config

## 3. Train model

You can train a model using the [full_train.sh](../bash_scripts/full_train.sh) script. The script is based on the guide
found [here (training)](https://opennmt.net/OpenNMT-py/examples/Translation.html#step-2-train-the-model) and
[here (translation and evaluation)](https://opennmt.net/OpenNMT-py/examples/Translation.html#step-3-translate-and-evaluate).

The script includes the following steps:
1. train a model
2. remove all the model files except for the one with the best score
3. translate the test/validation data
4. evaluate the translated data
5. pair the results into a file to easily view the translation quality
6. save the results to a history.tsv

### Sections used for configuration
- Script configs
- Data and vocab config
- Train config
