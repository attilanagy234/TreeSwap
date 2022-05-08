#!/bin/bash
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
utils_path=$(yq -r .utils_path config.yaml)

augmentation_active=$(yq  -r .multi_train.augmentation.active config.yaml)
multi_train_prefix=$(yq -r .multi_train.prefix config.yaml)
bare_train=$(yq -r .multi_train.bare config.yaml)

# python3 $utils_path/generate_multi_train_configs.py || exit 1
# sleep 1
# if $bare_train; then
#     # for folder in $(ls -l | grep "$multi_train_prefix" | grep "^d" | awk '{print $9}')
#     # do
#     #     pushd $folder && srun -p gpu --gres=mps $script_dir/1_build_vocab.sh &
#     #     popd
#     # done

#     if $augmentation_active; then
#         for folder in $(ls -l | grep "$multi_train_prefix" | grep "^d" | awk '{print $9}')
#         do
#             pushd $folder
#             # Remove bare train from config
#             # yq -i -y '.multi_train.bare |= false' config.yaml
#             # run the two commands sequentially on a node in the background
#             # job_name=aug-$(echo "$folder" | sed 's/[^-]*-[^-]*-\([^-]*\)-[^-]*-\(.*\)/\2/')
#             # srun -p gpu --exclusive --gres=mps --job-name=$job_name $script_dir/augment.sh &  
#             # python3 ../$utils_path/generate_multi_train_configs.py
#             # srun -p gpu --exclusive --gres=mps $script_dir/augment.sh && python3 ../$utils_path/generate_multi_train_configs.py &
#             job_name=$(echo "$folder" | sed 's/[^-]*-[^-]*-\([^-]*\)-[^-]*-\(.*\)/\1-\2/')
#             srun --exclusive -p gpu --gres=mps --job-name=$job_name $script_dir/full_train.sh &
#             popd
#         done
#     fi
# fi

if $bare_train && $augmentation_active; then
    for folder in $(ls -l | grep "$multi_train_prefix" | grep "^d" | awk '{print $9}')
    do
        pushd $folder
        for inner_folder in $(ls -l | grep "^d" | grep -v "run" | awk '{print $9}')
        do
            pushd $inner_folder
            job_name=$(echo "$folder" | sed 's/[^-]*-[^-]*-\([^-]*\)-[^-]*-\(.*\)/\1-\2/')
            srun -p gpu --gres=mps --exclusive --job-name=$job_name $script_dir/4_translate.sh && $script_dir/5_evaluate.sh &
            # yq -i -y '.test_src |= "/home1/hu-nmt/data/Hunglish2/low-resource/hunglish2-test.hu"' config.yaml
            # yq -i -y '.test_tgt |= "/home1/hu-nmt/data/Hunglish2/low-resource/hunglish2-test.en"' config.yaml
            # test_src: /home1/hu-nmt/data/Hunglish2/low-resource/hunglish2-test.en
            # test_tgt: /home1/hu-nmt/data/Hunglish2/low-resource/hunglish2-test.hu
            # srun --exclusive -p gpu --gres=mps --job-name=$job_name $script_dir/full_train.sh &
            sleep 1
            popd
        done
        popd
    done
else
    for folder in $(ls -l | grep "$multi_train_prefix" | grep "^d" | awk '{print $9}')
    do
        pushd $folder
        # yq -i -y '.test_src |= "/home1/hu-nmt/data/Hunglish2/low-resource/hunglish2-test.hu"' config.yaml
        # yq -i -y '.test_tgt |= "/home1/hu-nmt/data/Hunglish2/low-resource/hunglish2-test.en"' config.yaml
        job_name=$(echo "$folder" | sed 's/[^-]*-\([^-]*\)-[^-]*-\(.*\)/\1-\2/')
        # srun -p gpu --gres=mps --exclusive --job-name=$job_name $script_dir/4_translate.sh && $script_dir/5_evaluate.sh &
        srun --exclusive -p gpu --gres=mps --job-name=$job_name $script_dir/full_train.sh &
        popd
        # since all train runs write the same tsv
        # have a little gap between them so there is less chance for them to write the file the same time
        sleep 1
    done
fi
