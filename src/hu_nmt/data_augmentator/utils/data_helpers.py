from multiprocessing import Pool
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
import yaml
from dotmap import DotMap


def get_config_from_yaml(yaml_file):
    """
    Yaml config file to DotMap
    """
    with open(yaml_file, 'r') as config_file:
        config_yaml = yaml.load(config_file, Loader=yaml.FullLoader)
    # Using DotMap we will be able to reference nested parameters via attribute such as x.y instead of x['y']
    config = DotMap(config_yaml)
    return config


def parallelize_df_processing(df, func, num_cores, num_partitions):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def get_files_in_folder(folder_path):
    return [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
