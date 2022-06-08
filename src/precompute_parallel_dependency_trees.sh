#!/bin/bash

PYTHONPATH=. python -m hu_nmt.data_augmentator.entrypoints.precompute_parallel_dependency_trees "$@"