import argparse
import copy
import os

import yaml


def update_relative_paths(config: dict):
    for key, value in config.items():
        if type(value) == str:
            if value.startswith('../'):
                config[key] = '../' + value
        if type(value) == dict:
            update_relative_paths(value)


def dump_config_to_yaml(dir, config, file_name):
    if not os.path.exists(dir):
        os.mkdir(dir)
    with open(os.path.join(dir, file_name), 'w') as yaml_file:
        print(f'Creating yaml file {dir}/{file_name}')
        yaml.dump(config, yaml_file, default_flow_style=False, indent=4, sort_keys=False)


def swap_source_target_data(config, dir_name_with_prefix, aug_method=None):
    new_config = copy.deepcopy(config)
    for entry in new_config['data'].items():
        _, entry_value = entry
        entry_value['path_src'], entry_value['path_tgt'] = entry_value['path_tgt'], entry_value['path_src']

    new_config["test_src"], new_config["test_tgt"] = new_config["test_tgt"], new_config["test_src"]

    src_postfix = config['general']['src_postfix']
    tgt_postfix = config['general']['tgt_postfix']
    if aug_method:
        aug_method_dir_name = os.path.join(dir_name_with_prefix, f'{aug_method}-{tgt_postfix}{src_postfix}')
    else:
        aug_method_dir_name = dir_name_with_prefix
    dump_config_to_yaml(aug_method_dir_name, new_config, 'train_config.yaml')


def create_config(config, src_postfix, tgt_postfix, ratio=0, graph_method='base', threshold=None):
    new_config = copy.deepcopy(config)

    if ratio > 0:
        new_config['augmentation']['directory'] = os.path.join(config['multi_train']['data_directory'], 'preprocessed')
        new_config['augmentation']['augmented_folder_prefix'] = f'{graph_method}-{threshold}' if threshold else f'{graph_method}'
        new_config['augmentation']['preprocessor']['source_language'] = src_postfix
        new_config['augmentation']['preprocessor']['target_language'] = tgt_postfix
        new_config['augmentation']['augmentation_ratio'] = ratio
        new_config['augmentation']['augmentation_type'] = graph_method
        if threshold:
            new_config['augmentation']['similarity_threshold'] = threshold
            dir_name_with_prefix = f'{src_postfix}{tgt_postfix}-aug-{ratio}-{graph_method}-{threshold}'
        else:
            dir_name_with_prefix = f'{src_postfix}{tgt_postfix}-aug-{ratio}-{graph_method}'

        # save config to file
        update_relative_paths(new_config)
        dump_config_to_yaml(dir_name_with_prefix, new_config, 'aug_config.yaml')

        for aug_method in config['multi_train']['aug_method']:
            aug_method_dir_name = os.path.join(dir_name_with_prefix, f'{aug_method}-{src_postfix}{tgt_postfix}')

            aug_method_dir = 'obj_swapping' if aug_method == 'object' else 'subj_swapping'
            aug_dir = f'{graph_method}-{threshold}-{ratio}' if threshold else f'{graph_method}-{ratio}'
            augmented_full_path = os.path.join(new_config['augmentation']['directory'], aug_dir, aug_method_dir)

            new_config['data']['aug'] = dict()
            new_config['data']['aug']['path_src'] = os.path.join(augmented_full_path, f'{aug_method_dir}.src')
            new_config['data']['aug']['path_tgt'] = os.path.join(augmented_full_path, f'{aug_method_dir}.tgt')
            new_config['data']['aug']['transforms'] = ['sentencepiece']
            new_config['data']['aug']['weight'] = ratio

            update_relative_paths(new_config)
            dump_config_to_yaml(aug_method_dir_name, new_config, 'train_config.yaml')

            if config['multi_train']['backwards']:
                swap_source_target_data(new_config, dir_name_with_prefix, aug_method)

    else:
        dir_name_with_prefix = f'{src_postfix}{tgt_postfix}'
        dump_config_to_yaml(dir_name_with_prefix, new_config, 'train_config.yaml')
        if config['multi_train']['backwards']:
            dir_name_with_prefix = f'{tgt_postfix}{src_postfix}'
            swap_source_target_data(new_config, dir_name_with_prefix)


def generate_multi_train_configs(config_file_path: str):
    with open(config_file_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # check that multi train is enabled
    if config['multi_train']['active'] == False:
        raise RuntimeError('multi_train.active is not set to true, aborting...')

    # check for mandatory fields
    mandatory_fields = ['backwards', 'aug_method', 'graph_method', 'similarity_threshold', 'repeat', 'augmentation_ratio']
    for field in mandatory_fields:
        if field not in config['multi_train']:
            raise RuntimeError(f'multi_train.{field} field not set, aborting...')

    source_postfix = config['general']['src_postfix']
    target_postfix = config['general']['tgt_postfix']

    for ratio in config['multi_train']['augmentation_ratio']:
        if ratio == 0:
            create_config(config, source_postfix, target_postfix)
        else:
            for graph_method in config['multi_train']['graph_method']:
                if graph_method == 'base':
                    create_config(config, source_postfix, target_postfix, ratio, graph_method)
                else:
                    for threshold in config['multi_train']['similarity_threshold']:
                        create_config(config, source_postfix, target_postfix, ratio, graph_method, threshold)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--config-file', default='config.yaml', type=str, help='[REQUIRED] Config file to use.')
    args = parser.parse_args()

    generate_multi_train_configs(args.config_file)
