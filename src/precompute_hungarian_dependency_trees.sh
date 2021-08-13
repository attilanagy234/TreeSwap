#!/bin/bash

PYTHONPATH=. python -m hu_nmt.data_augmentator.entrypoints.precompute_hu_dependency_trees "$@"