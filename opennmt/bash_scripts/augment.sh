#!/bin/bash
multi_train_dir=$(yq -r .multi_train.directory config.yaml)
src_postfix=$(yq -r .multi_train.src_postfix config.yaml)
tgt_postfix=$(yq -r .multi_train.tgt_postfix config.yaml)
augmentation_dir=$(yq -r .multi_train.augmentation.directory config.yaml)
all_non_valid_src_paths=$(for dataset in $(yq ".data | keys" config.yaml | grep -v "valid" | grep ","); do dataset=$(echo $dataset | sed "s/\",*//g"); yq -r .data.$dataset.path_src config.yaml; done)
all_non_valid_tgt_paths=$(for dataset in $(yq ".data | keys" config.yaml | grep -v "valid" | grep ","); do dataset=$(echo $dataset | sed "s/\",*//g"); yq -r .data.$dataset.path_tgt config.yaml; done)
scripts_path=$(yq -r .multi_train.augmentation.scripts_path config.yaml)
preprocess_data_script=$(yq -r .multi_train.augmentation.preprocess_data_script config.yaml)

src_precompute_script=$(yq -r .multi_train.augmentation.src_precompute_script config.yaml)
tgt_precompute_script=$(yq -r .multi_train.augmentation.tgt_precompute_script config.yaml)
precompute_batch_size=$(yq -r .multi_train.augmentation.precompute_batch_size config.yaml)

augmentation_script=$(yq -r .multi_train.augmentation.augmentation_script config.yaml)
augmentation_ratio=$(yq -r .multi_train.augmentation.augmentation_ratio config.yaml)
translate_model=$(readlink -f $(yq -r .translate_model config.yaml))
reverse_translate_model=$(readlink -f $(yq -r .reverse_translate_model config.yaml))
src_subword_model=$(readlink -f $(yq -r .src_subword_model config.yaml))
augmented_folder_prefix=$(yq -r .multi_train.augmentation.augmented_folder_prefix config.yaml)
$filter_same_ancestor=$(yq -r .multi_train.augmentation.$filter_same_ancestor config.yaml)
$filter_same_pos_tag=$(yq -r .multi_train.augmentation.$filter_same_pos_tag config.yaml)
$filter_for_noun_tags=$(yq -r .multi_train.augmentation.$filter_for_noun_tags config.yaml)

absolute_augmentation_dir=$(readlink -f $augmentation_dir)

if [ ! -d $absolute_augmentation_dir/augmentation_input_data ]; then

    echo "Preprocessing data for augmentation"

    # Concat source training data
    tmp_src_train_file=tmp_src_train.txt
    rm -rf $tmp_src_train_file
    for f in $all_non_valid_src_paths
    do
        cat $f >> $tmp_src_train_file
    done
    absolute_tmp_src_train_path=$(readlink -f $tmp_src_train_file)

    # Concat target training data
    tmp_tgt_train_file=tmp_tgt_train.txt
    rm -rf $tmp_tgt_train_file
    for f in $all_non_valid_tgt_paths
    do
        cat $f >> $tmp_tgt_train_file
    done
    absolute_tmp_tgt_train_path=$(readlink -f $tmp_tgt_train_file)

    # Create preprocess.yaml
    # preprocessor_json_content=$(yq .multi_train.augmentation.preprocessor config.yaml)
    # echo "preprocessor: $preprocessor_json_content" | yq -y . > preprocess.yaml

    mkdir $absolute_augmentation_dir/augmentation_input_data

    pushd $scripts_path

    echo 'Running preprocessing'
    ./$preprocess_data_script \
    $absolute_tmp_src_train_path \
    $absolute_tmp_tgt_train_path \
    preprocess.yaml \
    $absolute_augmentation_dir/augmentation_input_data/preprocessed.$src_postfix \
    $absolute_augmentation_dir/augmentation_input_data/preprocessed.$tgt_postfix

    popd

    rm $absolute_tmp_src_train_path
    rm $absolute_tmp_tgt_train_path
fi

if [ ! -d $absolute_augmentation_dir/dependency_trees ]; then
    echo "Creating dependency trees"

    mkdir $absolute_augmentation_dir/dependency_trees
    mkdir $absolute_augmentation_dir/dependency_trees/$src_postfix
    mkdir $absolute_augmentation_dir/dependency_trees/$tgt_postfix

    pushd $scripts_path

    echo 'Running precompute on source'
    ./$src_precompute_script \
    $absolute_augmentation_dir/augmentation_input_data/preprocessed.$src_postfix \
    $absolute_augmentation_dir/dependency_trees/$src_postfix \
    $precompute_batch_size

    echo 'Running precompute on target'
    ./$tgt_precompute_script \
    $absolute_augmentation_dir/augmentation_input_data/preprocessed.$tgt_postfix \
    $absolute_augmentation_dir/dependency_trees/$tgt_postfix \
    $precompute_batch_size

    popd
fi

if [ ! -d  $absolute_augmentation_dir/$augmented_folder_prefix-$augmentation_ratio ]; then
    echo "Generating augmented sentences"
    mkdir $absolute_augmentation_dir/$augmented_folder_prefix-$augmentation_ratio

    pushd $scripts_path

    echo 'Running augmentation'
    ./$augmentation_script \
    $src_postfix $tgt_postfix \
    $absolute_augmentation_dir/dependency_trees/$src_postfix \
    $absolute_augmentation_dir/dependency_trees/$tgt_postfix \
    $absolute_augmentation_dir/$augmented_folder_prefix-$augmentation_ratio \
    $augmentation_ratio \
    --use_filters \
    --filter_quantile=0.25 \
    --src_model_path=$translate_model \
    --tgt_model_path=$reverse_translate_model \
    --sp_model_path=$src_subword_model \
    --filter_batch_size=512 \
    --filter_same_ancestor=$filter_same_ancestor \
    --filter_same_pos_tag=$filter_same_pos_tag \
    --filter_for_noun_tags=$filter_for_noun_tags

    popd
fi

# ./preprocess_data.sh /
# /home1/hu-nmt/data/Hunglish2/low-resource/training/hunglish2-75k/hunglish2-75k-train.en /
# /home1/hu-nmt/data/Hunglish2/low-resource/training/hunglish2-75k/hunglish2-75k-train.hu /
# preprocess.yaml /
# /home1/hu-nmt/data/Hunglish2/low-resource/training/hunglish2-75k/augmentation_input_data/preprocessed.en /
# /home1/hu-nmt/data/Hunglish2/low-resource/training/hunglish2-75k/augmentation_input_data/preprocessed.hu

# # srun ./precompute_english_dependency_trees.sh /home1/hu-nmt/data/Hunglish2/low-resource/training/hunglish2-75k/augmentation_input_data/preprocessed.en /home1/hu-nmt/data/Hunglish2/low-resource/training/hunglish2-75k/dependency_trees/en 10000
# ./precompute_english_dependency_trees.sh /
# /home1/hu-nmt/data/Hunglish2/low-resource/training/hunglish2-75k/augmentation_input_data/preprocessed.en /
# /home1/hu-nmt/data/Hunglish2/low-resource/training/hunglish2-75k/dependency_trees/en /
# 10000

# # srun -p fat2 --gres=mps ./precompute_hungarian_dependency_trees.sh /home1/hu-nmt/data/Hunglish2/low-resource/training/hunglish2-75k/augmentation_input_data/preprocessed.hu /home1/hu-nmt/data/Hunglish2/low-resource/training/hunglish2-75k/dependency_trees/hu 10000
# ./precompute_hungarian_dependency_trees.sh /
# /home1/hu-nmt/data/Hunglish2/low-resource/training/hunglish2-75k/augmentation_input_data/preprocessed.hu /
# /home1/hu-nmt/data/Hunglish2/low-resource/training/hunglish2-75k/dependency_trees/hu /
# 10000

# ./run_subject_object_augmentation.sh /
# en hu /
# /home1/hu-nmt/data/Hunglish2/low-resource/training/hunglish2-75k/dependency_trees/en /
# /home1/hu-nmt/data/Hunglish2/low-resource/training/hunglish2-75k/dependency_trees/hu /
# /home1/hu-nmt/data/Hunglish2/low-resource/training/hunglish2-75k/augmented /
# 0.5