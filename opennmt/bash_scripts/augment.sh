#!/bin/bash
augmentation_active=$(yq -r .augmentation.active config.yaml)

src_postfix=$(yq -r .general.src_postfix config.yaml)
tgt_postfix=$(yq -r .general.tgt_postfix config.yaml)
augmentation_dir=$(yq -r .augmentation.directory config.yaml)
all_non_valid_src_paths=$(for dataset in $(yq ".data | keys" config.yaml | grep -v "valid" | grep ","); do dataset=$(echo $dataset | sed "s/\",*//g"); yq -r .data.$dataset.path_src config.yaml; done)
all_non_valid_tgt_paths=$(for dataset in $(yq ".data | keys" config.yaml | grep -v "valid" | grep ","); do dataset=$(echo $dataset | sed "s/\",*//g"); yq -r .data.$dataset.path_tgt config.yaml; done)
scripts_path=$(yq -r .augmentation.scripts_path config.yaml)
preprocess_data_script=$(yq -r .augmentation.preprocess_data_script config.yaml)

src_precompute_script=$(yq -r .augmentation.src_precompute_script config.yaml)
tgt_precompute_script=$(yq -r .augmentation.tgt_precompute_script config.yaml)
precompute_batch_size=$(yq -r .augmentation.precompute_batch_size config.yaml)

augmentation_script=$(yq -r .augmentation.augmentation_script config.yaml)
augmentation_ratio=$(yq -r .augmentation.augmentation_ratio config.yaml)
augmented_folder_prefix=$(yq -r .augmentation.augmented_folder_prefix config.yaml)
filter_same_ancestor=$(yq -r .augmentation.filter_same_ancestor config.yaml)
filter_same_pos_tag=$(yq -r .augmentation.filter_same_pos_tag config.yaml)
filter_for_noun_tags=$(yq -r .augmentation.filter_for_noun_tags config.yaml)

absolute_augmentation_dir=$(readlink -f $augmentation_dir)

function preprocess_data_for_augmentation() {
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
    preprocessor_json_content=$(yq .augmentation.preprocessor config.yaml)
    echo "preprocessor: $preprocessor_json_content" | yq -y . > preprocess.yaml
    preprocess_yaml_path=$(readlink -f preprocess.yaml)

    mkdir -p $absolute_augmentation_dir/augmentation_input_data

    pushd $scripts_path

    echo 'Running preprocessing'
    ./$preprocess_data_script \
    $absolute_tmp_src_train_path \
    $absolute_tmp_tgt_train_path \
    $preprocess_yaml_path \
    $absolute_augmentation_dir/augmentation_input_data/preprocessed.$src_postfix \
    $absolute_augmentation_dir/augmentation_input_data/preprocessed.$tgt_postfix \
    

    popd

    rm $absolute_tmp_src_train_path
    rm $absolute_tmp_tgt_train_path
}

function create_dependency_trees() {
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
}

function augment() {
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
    --filter_same_ancestor=$filter_same_ancestor \
    --filter_same_pos_tag=$filter_same_pos_tag \
    --filter_for_noun_tags=$filter_for_noun_tags

    popd
}

if [ "$augmentation_active" == "true" ]; then
    if [ ! -d $absolute_augmentation_dir/augmentation_input_data ]; then
        preprocess_data_for_augmentation
    fi

    if [ ! -d $absolute_augmentation_dir/dependency_trees ]; then
        create_dependency_trees
    fi

    if [ ! -d  $absolute_augmentation_dir/$augmented_folder_prefix-$augmentation_ratio ]; then
        augment
    fi
else
    echo "Augmentation not enabled. Skipping..."
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