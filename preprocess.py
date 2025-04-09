import argparse
from config import get_config_from_file
from utils import get_dataloader

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help="yaml config file path")
    args = parser.parse_args()

    # get configurations from file
    config = get_config_from_file(config_file=args.config)
    # print('Configuration:')
    # print(config, end='\n\n')

    # get dataloader
    unsupervised_data = get_dataloader(config)
    unsupervised_data.preprocess()

