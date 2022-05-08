#!/bin/bash
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
utils_path=$(yq -r .utils_path config.yaml)

augmentation_active=$(yq  -r .multi_train.active config.yaml)
multi_train_prefix=$(yq -r .multi_train.prefix config.yaml)
bare_train=$(yq -r .multi_train.bare config.yaml)

# 1

# python3 $utils_path/generate_multi_train_configs.py || exit 1
# if $bare_train; then
#     for folder in $(ls -l | grep "$multi_train_prefix" | grep "^d" | awk '{print $9}')
#     do
#         pushd $folder
#         srun -p gpu --gres=mps $script_dir/1_build_vocab.sh &
#         sleep 1
#         popd
#     done

#     if $augmentation_active; then
#         for folder in $(ls -l | grep "$multi_train_prefix" | grep "^d" | awk '{print $9}')
#         do
#             pushd $folder
            
#             job_name=$(echo "$folder" | sed 's/[^-]*-[^-]*-\([^-]*\)-[^-]*-\(.*\)/\1-\2/')
#             srun --exclusive -p gpu --gres=mps --job-name=$job_name $script_dir/full_train.sh &

#             sleep 1
#             popd
#         done
#     fi
# fi

# 2

if $bare_train; then
    if $augmentation_active; then
        for folder in $(ls -l | grep "$multi_train_prefix" | grep "^d" | awk '{print $9}')
        do
            pushd $folder
            
            # dir_name=${PWD##*/}
            # translation_direction=$(echo "$dir_name" | sed 's/[^-]*-[^-]*-\([^-]*\)-[^-]*-\(.*\)/\1/')
            # src_lang=$(echo $translation_direction | cut -b -2)
            # tgt_lang=$(echo $translation_direction | cut -b 3-)
            # sample_size=$(echo "$dir_name" | sed 's/[^-]*-[^-]*-\([^-]*\)-[^-]*-\(.*\)/\2/')
            # destination_folder=$(ls -l .. | grep ^d | awk '{ print $9 }' | grep "$tgt_lang$src_lang" | grep "$sample_size")
            # echo "reverse_translate_model: ../$destination_folder/run/model.pt" >> config.yaml
            # sleep 1
            
            job_name=aug-$(echo "$folder" | sed 's/[^-]*-[^-]*-\([^-]*\)-[^-]*-\(.*\)/\2/')
            srun -p gpu --exclusive --gres=mps --job-name=$job_name $script_dir/augment.sh &
            sleep 0.5
            popd
        done
    fi
fi

# 3

# if $bare_train; then
#     if $augmentation_active; then
#         for folder in $(ls -l | grep "$multi_train_prefix" | grep "^d" | awk '{print $9}')
#         do
#             pushd $folder
            
#             # Remove bare train from config
#             yq -i -y '.multi_train.bare |= false' config.yaml
#             # Generate inner configs
#             python3 ../$utils_path/generate_multi_train_configs.py
#             sleep 1

#             job_name=$(echo "$folder" | sed 's/[^-]*-[^-]*-\([^-]*\)-[^-]*-\(.*\)/\1-\2/')

#             for inner_folder in $(ls -l | grep "^d" | grep -v run | awk '{print $9}')
#             do
#                 pushd $inner_folder

#                 # Add train_from parameter to config
#                 yq -i -y '.train_from |= "../run/model.pt"' config.yaml
                
#                 srun --exclusive -p gpu --gres=mps --job-name=$job_name $script_dir/full_train.sh &
#                 popd
#             done

#             popd
#         done
#     fi
# fi