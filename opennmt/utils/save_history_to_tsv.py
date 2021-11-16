import argparse
import collections
import os
import re
from datetime import datetime
from typing import Dict

import git
import pandas as pd
import yaml


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
    

def parse_yaml_to_df(yaml_path: str, tsv_path: str):
    with open(yaml_path, 'r') as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.FullLoader)

    flat_config = flatten_dict(config)

    if os.path.exists(tsv_path):
        df = pd.read_csv(tsv_path, sep='\t', header=0)
        df = df.append(flat_config, ignore_index=True)
    else:
        df = pd.DataFrame(flat_config)

    return df


def save_history(yaml_path: str, result_path: str, history_path: str, tsv_path: str):
    df = parse_yaml_to_df(yaml_path, tsv_path)

    df.loc[df.index[-1], 'date'] = datetime.today()
    df.loc[df.index[-1], 'history_path'] = history_path
    
    # get current git commit hash
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    df.loc[df.index[-1], 'git_hash'] = sha

    # parse bleu scores
    with open(result_path, 'r') as result_file:
        result_str = result_file.read()

    match = re.match('[^=]*=\ (\S+)\ ([^/]+)/([^/]+)/([^/]+)/(\S+)', result_str)

    df.loc[df.index[-1], 'bleu_score'] = match.group(1)
    for i in range(4):
        df.loc[df.index[-1], f'blue_score_{i+1}'] = match.group(i+2)

    df.to_csv(tsv_path, sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--yaml_path', type=str, required=True, help='[REQUIRED] Path to the yaml to parse to the tsv.')
    parser.add_argument('--result_path', type=str, required=True, help='[REQUIRED] Path to the final results.')
    parser.add_argument('--history_path', type=str, required=True, help='[REQUIRED] Path to the history directory where additional files are stored.')
    parser.add_argument('--tsv_path', type=str, required=True, help='[REQUIRED] Path to the tsv.')

    args = parser.parse_args()

    save_history(args.yaml_path, args.result_path, args.history_path, args.tsv_path)
