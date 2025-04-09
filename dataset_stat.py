import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from evaluation.post_process import post_processing, signal_to_psd, psd_to_hr
from evaluation.visualization import plot_psd, create_psd_figure, plot_hr_hist, save_figure
from matplotlib import pyplot as plt
from utils import get_dataloader
from config import get_config
from math import isclose

def dataset_psd_stats(config, dataloader):
    """
    collect psd and hr statistics from given dataloader.
    Args:
        config: config object
        dataloader: dataloader object
    Returns:
        psd_stats: psd statistics, shape (num_samples, psd_len)
        hr_stats: hr statistics, shape (num_samples, )
    """
    sbar = tqdm(dataloader, total=len(dataloader))
    # calculate HR and PSD using different bandpass values
    confidence_use_bandpass = config.CONFIDENCE_MODEL.BANDPASS
    confidence_low_pass = config.CONFIDENCE_MODEL.BANDPASS_LOW_FREQ
    confidence_high_pass = config.CONFIDENCE_MODEL.BANDPASS_HIGH_FREQ
    confidence_freq_resolution = config.CONFIDENCE_MODEL.FREQ_RESOLUTION
    inference_use_bandpass = config.POST_PROCESSING.BANDPASS
    inference_low_pass = config.POST_PROCESSING.BANDPASS_LOW_FREQ
    inference_high_pass = config.POST_PROCESSING.BANDPASS_HIGH_FREQ
    inference_freq_resolution = config.POST_PROCESSING.FREQ_RESOLUTION

    psd_len = len(np.arange(int(confidence_low_pass), int(confidence_high_pass+confidence_freq_resolution), int(confidence_freq_resolution)))
    psd_stats = []
    hr_stats = []

    # iterate over batch
    for batch_idx, batch in enumerate(sbar):
        # input shape: (batch_size, chunk_length, w, h, 9)
        # label shape: (batch_size, chunk_length, 3)
        input, label, filename, chunk_idx_local = batch
        batch_size = input.shape[0]

        # iterate over each sample in a batch
        for idx in range(batch_size):
            # input sample shape: (chunk_length, w, h, 9)
            # label sample shape: (chunk_length, 3)
            frames_sample, ppg_sample = input[idx].cpu().numpy(), label[idx].cpu().numpy()
            ppg_sample_raw = ppg_sample[:, 0]

            # psd stats for confidence model
            ppg_sample_confidence = post_processing(ppg_sample_raw, config.DATALOADER.FPS, diff_flag=False, use_bandpass=confidence_use_bandpass, low_pass=confidence_low_pass, high_pass=confidence_high_pass)
            freq_ppg_confidence, psd_ppg_confidence = signal_to_psd(ppg_sample_confidence, config.DATALOADER.FPS, freq_resolution=confidence_freq_resolution, low_pass=confidence_low_pass, high_pass=confidence_high_pass, interpolation=True)
            
            # hr stats for inference
            ppg_sample_inference = post_processing(ppg_sample_raw, config.DATALOADER.FPS, diff_flag=False, use_bandpass=inference_use_bandpass, low_pass=inference_low_pass, high_pass=inference_high_pass)
            freq_ppg_inference, psd_ppg_inference = signal_to_psd(ppg_sample_inference, config.DATALOADER.FPS, freq_resolution=inference_freq_resolution, low_pass=inference_low_pass, high_pass=inference_high_pass, interpolation=True)
            hr_label = psd_to_hr(freq_ppg_inference, psd_ppg_inference)

            psd_stats.append(psd_ppg_confidence)
            hr_stats.append(hr_label)

    return psd_stats, hr_stats

if __name__ == "__main__":
    # dataset_names = ['UBFC-rPPG', 'UBFC-Phys', 'iBVP', 'PURE'] # MAHNOB gt is ECG signal, not ppg signal
    dataset_names = ['PURE']

    """
    collect psd and hr statistics from all available datasets
    """
    for dataset_name in dataset_names:
        config_file = f"configs/preprocessing/{dataset_name}_preprocessing_72.yaml"
        config = get_config(config_file=config_file)

        # get dataloader
        unsupervised_data = get_dataloader(config)
        unsupervised_data.load_preprocessed_data_info()
        unsupervised_loader = DataLoader(
            dataset=unsupervised_data,
            num_workers=8,
            batch_size=1,
            shuffle=False,
        )

        # dataset psd statistics
        print(f"====== collect psd and hr statistics from {unsupervised_data.dataset_name} dataset ======")
        psd_stats, hr_stats = dataset_psd_stats(config, unsupervised_loader)
        os.makedirs(f'results/stats/', exist_ok=True)
        psd_stats_path = f'results/stats/{unsupervised_data.dataset_name}_psd_stats.npy'
        hr_stats_path = f'results/stats/{unsupervised_data.dataset_name}_hr_stats.npy'
        np.save(psd_stats_path, psd_stats)
        np.save(hr_stats_path, hr_stats)
        print(f"psd_stats shape: {np.shape(psd_stats)} saved to {psd_stats_path}")
        print(f"hr_stats shape: {np.shape(hr_stats)} saved to {hr_stats_path}")

    """
    conbine statistics from all datasets with PPG ground truth
    """
    # psd_stats_lst = []
    # hr_stats_lst = []
    # for dataset_name in dataset_names:
    #     psd_stats = np.load(f'results/stats/{dataset_name}_psd_stats.npy')
    #     hr_stats = np.load(f'results/stats/{dataset_name}_hr_stats.npy')
    #     psd_stats_lst.append(psd_stats)
    #     hr_stats_lst.append(hr_stats)
    # psd_stats_all = np.concatenate(psd_stats_lst, axis=0)
    # hr_stats_all = np.concatenate(hr_stats_lst, axis=0)
    # np.save(f'results/stats/all_psd_stats.npy', psd_stats_all)
    # np.save(f'results/stats/all_hr_stats.npy', hr_stats_all)

    """
    plot hr stats
    """
    hr_stats_all = []
    for dataset_name in dataset_names:
        hr_stats = np.load(f'results/stats/{dataset_name}_hr_stats.npy')
        plot_hr_hist(hr_stats, title=f"Histogram of HR for {dataset_name}")
        save_figure(f'results/stats/hr_hist_{dataset_name}.png')
        hr_stats_all.append(hr_stats)
    # conbine all hr stats
    hr_stats_all = np.concatenate(hr_stats_all, axis=0)
    plot_hr_hist(hr_stats_all, title=f"Histogram of HR for {dataset_names} datasets")
    save_figure(f'results/stats/hr_hist_all.png')
