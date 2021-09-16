import argparse
import os
from typing import List
import yaml
import copy
import subprocess


def get_dirs_at_path(path: str) -> List[str]:
    for dir, dirs, files in os.walk(path):
        if dir == path:
            return dirs
    
    raise RuntimeError('Did not find original directory while walking the path')


def generate_configs_for_translation(source_postfix: str, target_postfix: str, multi_train_dir: str, data_entry: dict, config: dict):
    # create configs in folders
    print(get_dirs_at_path(multi_train_dir))
    for dir_name in get_dirs_at_path(multi_train_dir):
        file_dict = {}
        
        # create dictionary from files
        file_names = os.listdir(os.path.join(multi_train_dir, dir_name))
        for file_name in file_names:
            file_parts = file_name.rpartition('.')
            name = file_parts[0]
            post_fix = file_parts[-1]

            if name not in file_dict:
                file_dict[name] = {}
            
            relative_file_path = os.path.join('..', multi_train_dir, dir_name, file_name)
            if post_fix == source_postfix:
                file_dict[name]['path_src'] = relative_file_path
            elif post_fix == target_postfix:
                file_dict[name]['path_tgt'] = relative_file_path
            else:
                raise RuntimeError('Did not match any postfix: {file_name}')
        
        # add transforms and weight to dict
        used_transforms = data_entry['transforms']
        for data_name, entries in file_dict.items():
            entries['transforms'] = used_transforms
            entries['weight'] = 1

        # create folder
        dir_name_with_prefix = source_postfix + target_postfix + '-' + dir_name
        os.mkdir(dir_name_with_prefix)

        # create config
        # new_config = config + extra_data_dict
        new_config = copy.deepcopy(config)
        new_config['translation_type'] = dir_name_with_prefix
        for corpus_name, corpus_data in file_dict.items():
            new_config['data'][corpus_name] = corpus_data
        with open(os.path.join(dir_name_with_prefix, 'config.yaml'), 'w') as yaml_file:
            print('Creating yaml file {dir_name_with_prefix}/config.yaml')
            yaml.dump(new_config, yaml_file, default_flow_style=None)


def generate_multi_train_configs(config_file_path: str):
    with open(config_file_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # check that multi train is enabled
    if config['multi_train'] == False:
        raise RuntimeError('multi_train is not set to true, aborting...')

    # check for mandatory fields
    mandatory_fields = ['multi_train_prefix', 'multi_train_dir']
    for field in mandatory_fields:
        if field not in config:
            raise RuntimeError(f'{field} field not set, aborting...')

    multi_train_prefix = config['multi_train_prefix']
    multi_train_dir = config['multi_train_dir']

    non_valid_data_entries = [(key, value) for key, value in config['data'].items() if key != 'valid']
    data_entry = non_valid_data_entries[0][1]
    source_postfix = data_entry['path_src'].rpartition('.')[-1]
    target_postfix = data_entry['path_tgt'].rpartition('.')[-1]

    generate_configs_for_translation(source_postfix, target_postfix, multi_train_dir, data_entry, config)

    # swap source and target datas
    for entry in config['data'].items():
        _, entry_value = entry
        entry_value['path_src'], entry_value['path_tgt'] = entry_value['path_tgt'], entry_value['path_src']
    generate_configs_for_translation(target_postfix, source_postfix, multi_train_dir, data_entry, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--config-file', default='config.yaml', type=str, help='[REQUIRED] Config file to use.')
    args = parser.parse_args()

    generate_multi_train_configs(args.config_file)
