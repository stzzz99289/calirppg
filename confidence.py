import numpy as np
import os
from config import get_config
from evaluation.confidence_model import StatistcsConfidenceModel
from utils import get_dataloader
from evaluation.visualization import plot_hr_hist, save_figure
from torch.utils.data import DataLoader
from tqdm import tqdm
from evaluation.post_process import post_processing, signal_to_psd, psd_to_hr
from evaluation.visualization import plot_psd, create_psd_figure
import matplotlib.pyplot as plt
from rppg_methods.unsupervised.unsupervised_predictor import sample_experiment

if __name__ == "__main__":
    config_file = "configs/confidence_model.yaml"
    config = get_config(config_file=config_file)
    mode = "fit_each" # ['fit_all', 'fit_each', 'test']

    if mode == "fit_each":
        """
        fit confidence model for each dataset
        """
        dataset_names = ['UBFC-rPPG', 'UBFC-Phys', 'iBVP', 'PURE']
        for dataset_name in dataset_names:
            # load stats
            psd_stats = np.load(f'results/stats/{dataset_name}_psd_stats.npy')
            hr_stats = np.load(f'results/stats/{dataset_name}_hr_stats.npy')
            print(f"fit confidence model on {len(psd_stats)} samples for {dataset_name} dataset")

            # fit confidence model
            confidence_model = StatistcsConfidenceModel(config)
            confidence_model.fit(psd_stats, hr_stats)
            confidence_model.save_statistics_plots(save_path=f'results/psd_distribution/{dataset_name}/gaussian')

            """
            save confidence model
            """
            confidence_model_path = f"results/confidence_model/statistics_{dataset_name}"
            print(f"save confidence model to {confidence_model_path}")
            confidence_model.save(confidence_model_path)
    

    elif mode == "fit_all":
        """
        fit confidence model for combination of all datasets
        """
        psd_stats_all = np.load(f'results/stats/all_psd_stats.npy')
        hr_stats_all = np.load(f'results/stats/all_hr_stats.npy')
        print(f"fit confidence model on {len(psd_stats_all)} samples for all datasets")
        confidence_model = StatistcsConfidenceModel(config)
        confidence_model.fit(psd_stats_all, hr_stats_all)
        confidence_model.save_statistics_plots(save_path=f'results/stats/all/refpsd_multivariate_gaussian')

        """
        save confidence model
        """
        confidence_model_path = "results/confidence_model/statistics_all"
        print(f"save confidence model to {confidence_model_path}")
        confidence_model.save(confidence_model_path)


    elif mode == "test":
        """
        test confidence model prediction
        """
        confidence_model = StatistcsConfidenceModel(config)
        confidence_model.load("results/confidence_model/statistics")
        dataset_name = "PURE"
        config = get_config(config_file=f"configs/{dataset_name}_inference.yaml")
        unsupervised_data = get_dataloader(config)
        unsupervised_data.load_preprocessed_data_info()
        unsupervised_loader = DataLoader(
            dataset=unsupervised_data,
            num_workers=1,
            batch_size=1,
            shuffle=False,
        )
        sample_experiment(config, unsupervised_loader, confidence_model=confidence_model)
