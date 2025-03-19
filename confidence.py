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
    """
    fit confidence model for each dataset
    """
    # dataset_names = ['UBFC-rPPG', 'UBFC-Phys', 'iBVP', 'PURE', 'MAHNOB']
    # for dataset_name in dataset_names:
    #     # load stats
    #     psd_stats = np.load(f'results/stats/{dataset_name}_psd_stats.npy')
    #     hr_stats = np.load(f'results/stats/{dataset_name}_hr_stats.npy')
    #     print(f"fit confidence model on {len(psd_stats)} samples for {dataset_name} dataset")

    #     # fit confidence model
    #     confidence_model = StatistcsConfidenceModel(config)
    #     confidence_model.fit(psd_stats, hr_stats)
    #     confidence_model.save_statistics_plots(save_path=f'results/stats/{dataset_name}/refpsd_raw', plot_smoothed=False)
    #     confidence_model.save_statistics_plots(save_path=f'results/stats/{dataset_name}/refpsd_smoothed', plot_smoothed=True)
    
    """
    conbine statistics from all datasets with PPG ground truth
    """
    # dataset_names = ['UBFC-rPPG', 'UBFC-Phys', 'iBVP', 'PURE'] # MAHNOB gt is ECG signal, not ppg signal
    # psd_stats_lst = []
    # hr_stats_lst = []
    # for dataset_name in dataset_names:
    #     psd_stats = np.load(f'results/stats/{dataset_name}_psd_stats.npy')
    #     hr_stats = np.load(f'results/stats/{dataset_name}_hr_stats.npy')
    #     psd_stats_lst.append(psd_stats)
    #     hr_stats_lst.append(hr_stats)
    # psd_stats_all = np.concatenate(psd_stats_lst, axis=0)
    # hr_stats_all = np.concatenate(hr_stats_lst, axis=0)

    """
    fit confidence model for combination of all datasets
    """
    # print(f"fit confidence model on {len(psd_stats_all)} samples for {dataset_names} datasets")
    # confidence_model = StatistcsConfidenceModel(config)
    # confidence_model.fit(psd_stats_all, hr_stats_all)
    # confidence_model.save_statistics_plots(save_path=f'results/stats/all/refpsd_raw', plot_smoothed=False)
    # confidence_model.save_statistics_plots(save_path=f'results/stats/all/refpsd_smoothed', plot_smoothed=True)

    """
    save confidence model
    """
    # confidence_model_path = "results/confidence_model/statistics"
    # print(f"save confidence model to {confidence_model_path}")
    # confidence_model.save(confidence_model_path)

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
