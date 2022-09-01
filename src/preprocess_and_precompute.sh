#!/bin/bash

PYTHONPATH=. python -m hu_nmt.data_augmentator.entrypoints.preprocess_and_precompute_dep_trees "$@"