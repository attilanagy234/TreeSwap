#!/bin/bash
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
utils_path=$(yq -r .utils_path config.yaml)

augmentation_active=$(yq  -r .multi_train.active config.yaml)
bare_train=$(yq -r .multi_train.bare config.yaml)

# 1 generating config files

 python3 $utils_path/generate_multi_train_configs_a100.py || exit 1

# 2 running augmentation

if $bare_train; then
    if $augmentation_active; then
        for config_file in $(find ~+ -name aug_config.yaml)
        do
            pushd $(dirname $config_file)

            $script_dir/augment.sh &
            sleep 0.5
            popd
        done
    fi
fi

# 3 training

 if $bare_train; then
     if $augmentation_active; then
         for config_file in $(find ~+ -name train_config.yaml)
         do
             pushd $(dirname $config_file)

             $script_dir/full_train.sh &
             sleep 0.5

             popd
         done
     fi
 fi