import os
import yaml
from schema import SchemaError
from src.utils.schema import CONFIG_SCHEMA_CLUSTERING, CONFIG_SCHEMA_GASTEN


def read_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
        # add paths
        for rel_path in ['out-dir', 'data-dir', 'fid-stats-path', 'test-noise'] :
            config[rel_path] = os.environ['FILESDIR'] + '/' + config[rel_path]
        config['train']['step-2']['classifier'] = [(os.environ['FILESDIR'] + '/' + rel_path) for rel_path in config['train']['step-2']['classifier']]
    try:
        CONFIG_SCHEMA_GASTEN.validate(config)
    except SchemaError as se:
        raise se

    if "run-seeds" in config and len(config["run-seeds"]) != config["num-runs"]:
        print("Number of seeds must be equal to number of runs")
        exit(-1)

    if "run-seeds" in config["train"]["step-2"] and \
            len(config["train"]["step-2"]["run-seeds"]) != config["num-runs"]:
        print("Number of mod_gan seeds must be equal to number of runs")
        exit(-1)

    return config


def read_config_clustering(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
        # add paths
        for rel_path in ['data', 'fid-stats', 'clustering']:
            config['dir'][rel_path] = os.environ['FILESDIR'] + '/' + config['dir'][rel_path]
        config['gasten']['classifier'] = [(os.environ['FILESDIR'] + '/' + rel_path) for rel_path in config['gasten']['classifier']]
    try:
        CONFIG_SCHEMA_CLUSTERING.validate(config)
    except SchemaError as se:
        raise se
    return config
