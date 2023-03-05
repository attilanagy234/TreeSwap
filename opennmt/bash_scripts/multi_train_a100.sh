#!/bin/bash
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
utils_path=$(yq -r .utils_path config.yaml)

augmentation_active=$(yq  -r .multi_train.active config.yaml)
repeat=$(yq -r .multi_train.repeat config.yaml)

# 1 generating config files
 if $generate_configs; then
    python3 $utils_path/generate_multi_train_configs_a100.py || exit 1
 fi

# 2 running augmentation

  if $augmentation_active; then
      for config_file in $(find ~+ -name aug_config.yaml)
      do
          pushd $(dirname $config_file)

          $script_dir/augment.sh &
          sleep 0.5
          popd
      done
  fi

# 3 training

   if $augmentation_active; then
       for config_file in $(find ~+ -name train_config.yaml)
       do
           pushd $(dirname $config_file)

           for i in $(seq 1 $repeat)
           do
             if [ ! -d  run-$i ]; then
                $script_dir/full_train.sh &
                mv run run-$i
                sleep 0.5
             else
               echo "Training already done. Skipping..."
             fi

           done

           popd
       done
   fi