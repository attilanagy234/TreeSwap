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
