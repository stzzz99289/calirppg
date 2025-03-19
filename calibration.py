import argparse
from config import get_config
from dataloader.ibvp import iBVPLoader

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help="yaml config file path")
    args = parser.parse_args()

    # get configurations from file
    config = get_config(args)
    print('Configuration:')
    print(config, end='\n\n')

    # dataloader
    ibvploader = iBVPLoader(config)
    