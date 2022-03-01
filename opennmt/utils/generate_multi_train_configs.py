import argparse
import os
from typing import List
import yaml
import copy
import subprocess
from pprint import pprint


def get_dirs_at_path(path: str) -> List[str]:
    for dir, dirs, files in os.walk(path):
        if dir == path:
            return dirs
    
    raise RuntimeError('Did not find original directory while walking the path')


def wccount(filename) -> int:
    """
    Count the number of lines in a given file.
    """
    out = subprocess.Popen(['wc', '-l', filename],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT
                         ).communicate()[0]
    return int(out.partition(b' ')[0])


def update_relative_paths(config: dict):
    for key, value in config.items():
        if type(value) == str:
            if value.startswith('../'):
                config[key] = '../' + value
        if type(value) == dict:
            update_relative_paths(value)


def generate_config_for_translation(source_postfix: str, target_postfix: str, config: dict):
    multi_train_dir = config['multi_train']['directory']
    
    # create configs in folders
    for dir_name in get_dirs_at_path(multi_train_dir):
        file_dict = {}
        
        # create dictionary from files
        file_names = os.listdir(os.path.join(multi_train_dir, dir_name))
        for file_name in file_names:
            if os.path.isfile(os.path.join(multi_train_dir, dir_name, file_name)):
                file_parts = file_name.rpartition('.')
                name = file_parts[0].replace("-", "_")
                post_fix = file_parts[-1]
                
                if config['multi_train']['bare'] and 'valid' in file_name:
                    name = 'valid'
                    
                relative_file_path = os.path.join(multi_train_dir, dir_name, file_name)
                if post_fix == source_postfix:
                    if name not in file_dict:
                        file_dict[name] = {}
                    file_dict[name]['path_src'] = relative_file_path
                elif post_fix == target_postfix:
                    if name not in file_dict:
                        file_dict[name] = {}
                    file_dict[name]['path_tgt'] = relative_file_path
                else:
                    if post_fix != 'sp':
                        raise RuntimeError(f'Did not match any postfix: {file_name}')

        # create folder
        dir_name_with_prefix = config['multi_train']['prefix'] + '-' + source_postfix + target_postfix + '-' + dir_name
        if not os.path.exists(dir_name_with_prefix):
            os.mkdir(dir_name_with_prefix)

        # create config
        # new_config = config + extra_data_dict
        new_config = copy.deepcopy(config)
        new_config['multi_train']['src_postfix'] = source_postfix
        new_config['multi_train']['tgt_postfix'] = target_postfix
        new_config['translation_type'] = dir_name_with_prefix
        new_config['multi_train']['directory'] = os.path.join(multi_train_dir, dir_name)

        if config['multi_train']['bare']:
            new_config['data'] = {}
        for corpus_name, corpus_data in file_dict.items():
            new_config['data'][corpus_name] = corpus_data

        # set multi_train dir for augmentation case
        if config['multi_train']['augmentation']['active'] and config['multi_train']['bare']:
            augmentation_ratio = new_config['multi_train']['augmentation']['augmentation_ratio']
            new_config['multi_train']['directory'] = os.path.join(multi_train_dir, dir_name, f"{config['multi_train']['augmentation']['augmented_folder_prefix']}-{augmentation_ratio}")
            new_config['multi_train']['augmentation']['directory'] = os.path.join(multi_train_dir, dir_name)

        # calculate number of steps if needed
        if 'epochs' in config['multi_train']:
            if config['batch_type'] == 'tokens':
                max_token_count = config['multi_train']['max_token_count']
                batch_size = config['batch_size']

                number_of_sentences = 0
                all_non_valid_source_files = [corpus_data['path_src'] for corpus_name, corpus_data in new_config['data'].items() if corpus_name != 'valid']
                for file in all_non_valid_source_files:
                    number_of_sentences += wccount(file)

                steps_per_epoch = int((number_of_sentences * max_token_count) / batch_size)
                new_config['train_steps'] = config['multi_train']['epochs'] * steps_per_epoch
                new_config['valid_steps'] = steps_per_epoch
                new_config['save_checkpoint_steps'] = steps_per_epoch
                new_config['warmup_steps'] = 2 * steps_per_epoch
            elif config['batch_type'] == 'sent':
                pass
        
        # Set vocab paths if bare
        if config['multi_train']['bare']:
            vocabs_path = os.path.join(multi_train_dir, dir_name, 'vocabs')
            if not os.path.exists(vocabs_path):
                os.mkdir(vocabs_path)
            new_config['save_data'] = vocabs_path
            new_config['src_vocab'] = os.path.join(vocabs_path, f'vocab.{source_postfix}{target_postfix}')

            sp_models_path = os.path.join(multi_train_dir, dir_name, 'sp_models')
            if not os.path.exists(sp_models_path):
                os.mkdir(sp_models_path)
            new_config['sp_models_path'] = sp_models_path
            new_config['src_subword_model'] = os.path.join(sp_models_path, f'spm_{source_postfix}{target_postfix}.model')

        # save config to file
        update_relative_paths(new_config)
        with open(os.path.join(dir_name_with_prefix, 'config.yaml'), 'w') as yaml_file:
            print(f'Creating yaml file {dir_name_with_prefix}/config.yaml')
            yaml.dump(new_config, yaml_file, default_flow_style=None)


def generate_multi_train_configs(config_file_path: str):
    with open(config_file_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # check that multi train is enabled
    if config['multi_train']['active'] == False:
        raise RuntimeError('multi_train.active is not set to true, aborting...')

    # check for mandatory fields
    mandatory_fields = ['prefix', 'directory', 'src_postfix', 'tgt_postfix', 'bare']
    if 'epochs' in config['multi_train']:
        mandatory_fields.append('max_token_count')
    for field in mandatory_fields:
        if field not in config['multi_train']:
            raise RuntimeError(f'multi_train.{field} field not set, aborting...')

    source_postfix = config['multi_train']['src_postfix']
    target_postfix = config['multi_train']['tgt_postfix']

    generate_config_for_translation(source_postfix, target_postfix, config)

    if not config['multi_train']['augmentation']['active'] or (config['multi_train']['augmentation']['active'] and config['multi_train']['bare']):
        # swap source and target datas
        for entry in config['data'].items():
            _, entry_value = entry
            entry_value['path_src'], entry_value['path_tgt'] = entry_value['path_tgt'], entry_value['path_src']
        generate_config_for_translation(target_postfix, source_postfix, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--config-file', default='config.yaml', type=str, help='[REQUIRED] Config file to use.')
    args = parser.parse_args()

    generate_multi_train_configs(args.config_file)
