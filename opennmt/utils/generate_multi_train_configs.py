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


def generate_configs_for_translation(source_postfix: str, target_postfix: str, config: dict):
    multi_train_dir = config['multi_train_dir']
    
    # create configs in folders
    for dir_name in get_dirs_at_path(multi_train_dir):
        file_dict = {}
        
        # create dictionary from files
        file_names = os.listdir(os.path.join(multi_train_dir, dir_name))
        for file_name in file_names:
            if os.path.isfile(os.path.join(multi_train_dir, dir_name, file_name)):
                file_parts = file_name.rpartition('.')
                name = file_parts[0]
                post_fix = file_parts[-1]
                
                if config['bare_multi_train'] and 'valid' in file_name:
                    name = 'valid'

                if name not in file_dict:
                    file_dict[name] = {}
                    
                relative_file_path = os.path.join('..', multi_train_dir, dir_name, file_name)
                if post_fix == source_postfix:
                    file_dict[name]['path_src'] = relative_file_path
                elif post_fix == target_postfix:
                    file_dict[name]['path_tgt'] = relative_file_path
                else:
                    raise RuntimeError('Did not match any postfix: {file_name}')

        # create folder
        dir_name_with_prefix = config['multi_train_prefix'] + '-' + source_postfix + target_postfix + '-' + dir_name
        os.mkdir(dir_name_with_prefix)

        # create config
        # new_config = config + extra_data_dict
        new_config = copy.deepcopy(config)
        new_config['translation_type'] = dir_name_with_prefix
        if config['bare_multi_train']:
            new_config['data'] = {}
        for corpus_name, corpus_data in file_dict.items():
            new_config['data'][corpus_name] = corpus_data
        with open(os.path.join(dir_name_with_prefix, 'config.yaml'), 'w') as yaml_file:
            print(f'Creating yaml file {dir_name_with_prefix}/config.yaml')
            yaml.dump(new_config, yaml_file, default_flow_style=None)


def generate_multi_train_configs(config_file_path: str):
    with open(config_file_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # check that multi train is enabled
    if config['multi_train'] == False:
        raise RuntimeError('multi_train is not set to true, aborting...')

    # check for mandatory fields
    mandatory_fields = ['multi_train_prefix', 'multi_train_dir', 'src_postfix', 'tgt_postfix', 'bare_multi_train']
    for field in mandatory_fields:
        if field not in config:
            raise RuntimeError(f'{field} field not set, aborting...')

    source_postfix = config['src_postfix']
    target_postfix = config['tgt_postfix']

    generate_configs_for_translation(source_postfix, target_postfix, config)

    # swap source and target datas
    for entry in config['data'].items():
        _, entry_value = entry
        entry_value['path_src'], entry_value['path_tgt'] = entry_value['path_tgt'], entry_value['path_src']
    generate_configs_for_translation(target_postfix, source_postfix, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--config-file', default='config.yaml', type=str, help='[REQUIRED] Config file to use.')
    args = parser.parse_args()

    generate_multi_train_configs(args.config_file)
