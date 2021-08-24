#!/bin/bash
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
utils_path_from_multi_train=$(grep 'utils_path_from_multi_train:' config.yaml | awk '{ print $2 }')

python3 $utils_path_from_multi_train/generate_multi_train_configs.py || exit 1
for folder in $(ls -l | grep "^d" | awk '{print $9}')
do
    pushd $folder && $script_dir/full_train.sh &
    popd
done