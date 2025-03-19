import argparse
import os
import pandas as pd
import numpy as np
from rppg_methods.unsupervised.unsupervised_predictor import unsupervised_predict, sample_experiment
from evaluation.confidence_model import StatistcsConfidenceModel
from config import get_config
from utils import get_dataloader
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help="yaml config file path")
    args = parser.parse_args()

    # get configurations from file
    config = get_config(config_file=args.config)
    print('Configuration:')
    print(config, end='\n\n')

    # get dataloader
    unsupervised_data = get_dataloader(config)
    unsupervised_data.load_preprocessed_data_info()
    unsupervised_loader = DataLoader(
        dataset=unsupervised_data,
        num_workers=1,
        batch_size=1,
        shuffle=False,
    )

    # inference experiment for on sample
    # sample_experiment(config, unsupervised_loader)

    # load confidence model
    confidence_model = StatistcsConfidenceModel(get_config(config_file="configs/confidence_model.yaml"))
    confidence_model.load(load_path="results/confidence_model/statistics")

    # inference
    global_metrics_lst = []
    unsupervised_method_names = ['POS', 'CHROM', 'ICA', 'GREEN', 'LGI', 'PBV', 'OMIT']
    for method_name in unsupervised_method_names:
        global_metrics = unsupervised_predict(config, unsupervised_loader, method_name=method_name, confidence_model=confidence_model)
        global_metrics_lst.append(global_metrics)

    # # save global metrics
    # global_metrics_df = pd.DataFrame(global_metrics_lst)
    # global_metrics_dir = f'results/{unsupervised_data.dataset_name}'
    # os.makedirs(global_metrics_dir, exist_ok=True)
    # global_metrics_df.to_csv(f'{global_metrics_dir}/global_metrics.csv', index=False)
