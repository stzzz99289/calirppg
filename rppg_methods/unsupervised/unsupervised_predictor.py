"""Unsupervised learning methods including POS, GREEN, CHROME, ICA, LGI and PBV."""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from evaluation.post_process import post_processing, signal_to_psd, psd_to_hr
from evaluation.visualization import create_psd_figure, plot_psd, save_figure, plot_psd_statistics
from evaluation.metric import compute_metric_avg_std, compute_mae, compute_rmse, compute_mape, compute_pearson, compute_macc, compute_snr
from rppg_methods.unsupervised.methods.CHROME_DEHAAN import CHROME_DEHAAN
from rppg_methods.unsupervised.methods.GREEN import GREEN
from rppg_methods.unsupervised.methods.ICA_POH import ICA_POH
from rppg_methods.unsupervised.methods.LGI import LGI
from rppg_methods.unsupervised.methods.PBV import PBV
from rppg_methods.unsupervised.methods.POS_WANG import POS_WANG
from rppg_methods.unsupervised.methods.OMIT import OMIT
from tqdm import tqdm
# from evaluation.BlandAltmanPy import BlandAltman

def sample_experiment(config, dataloader, confidence_model=None, unsupervised_method_names=['POS', 'CHROM', 'ICA', 'GREEN', 'LGI', 'PBV', 'OMIT']):
    fps = config.DATALOADER.FPS
    inference_low_pass = config.POST_PROCESSING.BANDPASS_LOW_FREQ
    inference_high_pass = config.POST_PROCESSING.BANDPASS_HIGH_FREQ
    inference_freq_resolution = config.POST_PROCESSING.FREQ_RESOLUTION
    confidence_low_pass = config.CONFIDENCE_MODEL.BANDPASS_LOW_FREQ
    confidence_high_pass = config.CONFIDENCE_MODEL.BANDPASS_HIGH_FREQ
    confidence_freq_resolution = config.CONFIDENCE_MODEL.FREQ_RESOLUTION

    # get one data sample
    sbar = tqdm(dataloader, total=len(dataloader))
    for batch_idx, batch in enumerate(sbar):
        input, label, filename, chunk_idx_local = batch
        frames_sample, ppg_sample = input[0].cpu().numpy(), label[0].cpu().numpy()

        """
        inference plot
        """
        # label
        ppg_sample = post_processing(ppg_sample, fps, low_pass=inference_low_pass, high_pass=inference_high_pass)
        freq_ppg, psd_ppg = signal_to_psd(ppg_sample, fps, freq_resolution=inference_freq_resolution, low_pass=inference_low_pass, high_pass=inference_high_pass, interpolation=True)
        hr_label = psd_to_hr(freq_ppg, psd_ppg)
        sbar.set_description(f"hr_label = {hr_label:3.0f} bpm")
        # plot
        create_psd_figure(title=f"gt: {hr_label} bpm", xlim=[inference_low_pass, inference_high_pass], xticks=np.arange(inference_low_pass, inference_high_pass+1, inference_freq_resolution*2))
        plot_psd(freq_ppg, psd_ppg, label="gt ppg")
        plt.axvline(hr_label, linestyle='--', color='red', alpha=0.8)
        for method_name in unsupervised_method_names:
            rppg = frames2rPPG_unsupervised(frames_sample, fps, method_name)
            rppg = post_processing(rppg, fps, low_pass=inference_low_pass, high_pass=inference_high_pass)
            freq_rppg, psd_rppg = signal_to_psd(rppg, fps, freq_resolution=inference_freq_resolution, low_pass=inference_low_pass, high_pass=inference_high_pass, interpolation=True)
            plot_psd(freq_rppg, psd_rppg, label=f"{method_name}", alpha=0.3)
        save_figure(f"gt_psd_plot_{hr_label}.png")

        """
        confidence plot
        """
        for method_name in unsupervised_method_names:
            create_psd_figure(title=f"gt: {hr_label} bpm", xlim=[confidence_low_pass, confidence_high_pass], xticks=np.arange(confidence_low_pass, confidence_high_pass+1, confidence_freq_resolution*2))
            plt.axvline(hr_label, linestyle='--', color='red', alpha=0.8)

            rppg = frames2rPPG_unsupervised(frames_sample, fps, method_name)
            rppg = post_processing(rppg, fps, low_pass=confidence_low_pass, high_pass=confidence_high_pass)
            freq_rppg, psd_rppg = signal_to_psd(rppg, fps, freq_resolution=confidence_freq_resolution, low_pass=confidence_low_pass, high_pass=confidence_high_pass, interpolation=True)
            hr_pred = psd_to_hr(freq_rppg, psd_rppg)
            if confidence_model is not None:
                confidence = confidence_model.predict(hr_pred, freq_rppg, psd_rppg)
                plot_psd(freq_rppg, psd_rppg, label=f"hr_pred={hr_pred:.0f} bpm, confidence={confidence:.2f}")
                rppg_hr_idx = np.argmin(np.abs(confidence_model.hr_values - hr_pred))
                ref_psd_mean = confidence_model.psd_statistics_smoothed["psd_means"][rppg_hr_idx]
                ref_psd_std = confidence_model.psd_statistics_smoothed["psd_stds"][rppg_hr_idx]
                plot_psd_statistics(freq_rppg, ref_psd_mean, ref_psd_std, label="reference psd", alpha=0.5)
            save_figure(f"confidence_plot_{hr_label}_{method_name}.png")
        break
           
def unsupervised_predict(config, dataloader, method_name, confidence_model=None):
    sbar = tqdm(dataloader, total=len(dataloader))

    # bandpass for inference (calculate hr)
    use_bandpass = config.POST_PROCESSING.BANDPASS
    low_pass = config.POST_PROCESSING.BANDPASS_LOW_FREQ
    high_pass = config.POST_PROCESSING.BANDPASS_HIGH_FREQ
    freq_resolution = config.POST_PROCESSING.FREQ_RESOLUTION

    # bandpass for confidence model (we need wider range to capture second harmonic information)
    confidence_low_pass = config.CONFIDENCE_MODEL.BANDPASS_LOW_FREQ
    confidence_high_pass = config.CONFIDENCE_MODEL.BANDPASS_HIGH_FREQ
    confidence_freq_resolution = config.CONFIDENCE_MODEL.FREQ_RESOLUTION

    hr_pred_lst = []
    hr_label_lst = []
    snr_lst = []
    macc_lst = []
    local_metrics = []

    # iterate over batch
    for batch_idx, batch in enumerate(sbar):
        # input shape: (batch_size, chunk_length, w, h, 3)
        # label shape: (batch_size, chunk_length, )
        input, label, filename, chunk_idx_local = batch
        batch_size = input.shape[0]

        # iterate over each sample in a batch
        for idx in range(batch_size):
            # input sample shape: (chunk_length, w, h, 3)
            # label sample shape: (chunk_length, )
            frames_sample, ppg_sample = input[idx].cpu().numpy(), label[idx].cpu().numpy()
            filename_sample, chunk_idx_local_sample = filename[idx], int(chunk_idx_local[idx])

            # frames to rPPG
            rppg = frames2rPPG_unsupervised(frames_sample, config.DATALOADER.FPS, method_name)

            # post-processing
            rppg = post_processing(rppg, config.DATALOADER.FPS, diff_flag=False, use_bandpass=use_bandpass, low_pass=low_pass, high_pass=high_pass)
            rppg_confidence = post_processing(rppg, config.DATALOADER.FPS, diff_flag=False, use_bandpass=use_bandpass, low_pass=confidence_low_pass, high_pass=confidence_high_pass)
            ppg_sample = post_processing(ppg_sample, config.DATALOADER.FPS, diff_flag=False, use_bandpass=use_bandpass, low_pass=low_pass, high_pass=high_pass)

            # PPG to PSD
            freq_rppg, psd_rppg = signal_to_psd(rppg, config.DATALOADER.FPS, freq_resolution=freq_resolution, low_pass=low_pass, high_pass=high_pass, interpolation=True)
            freq_rppg_confidence, psd_rppg_confidence = signal_to_psd(rppg_confidence, config.DATALOADER.FPS, freq_resolution=confidence_freq_resolution, low_pass=confidence_low_pass, high_pass=confidence_high_pass, interpolation=True)
            freq_ppg, psd_ppg = signal_to_psd(ppg_sample, config.DATALOADER.FPS, freq_resolution=freq_resolution, low_pass=low_pass, high_pass=high_pass, interpolation=True)

            # PSD to HR
            hr_pred = psd_to_hr(freq_rppg, psd_rppg)
            hr_label = psd_to_hr(freq_ppg, psd_ppg)

            # calculate confidence score
            if confidence_model is not None:
                confidence = confidence_model.predict(hr_pred, freq_rppg_confidence, psd_rppg_confidence)
            else:
                confidence = None

            # evaluation (rPPG vs PPG label)
            macc = compute_macc(rppg, ppg_sample)
            snr = compute_snr(rppg, hr_label)
            hr_pred_lst.append(hr_pred)
            hr_label_lst.append(hr_label)
            snr_lst.append(snr)
            macc_lst.append(macc)
            local_metrics.append({
                'filename': filename_sample, 
                'chunk_idx_local': chunk_idx_local_sample, 
                'hr_pred': hr_pred, 
                'hr_label': hr_label, 
                'snr': snr, 
                'macc': macc,
                'confidence': confidence,
                'abs_hr_error': np.abs(hr_pred - hr_label)
                })
    hr_pred_lst = np.array(hr_pred_lst)
    hr_label_lst = np.array(hr_label_lst)
    snr_lst = np.array(snr_lst)
    macc_lst = np.array(macc_lst)

    # save local metrics
    dataset_name = config.DATALOADER.DATASET
    local_metrics_dir = f'results/{dataset_name}'
    os.makedirs(local_metrics_dir, exist_ok=True)
    local_metrics_df = pd.DataFrame(local_metrics)
    local_metrics_df.to_csv(f'{local_metrics_dir}/{method_name}_metrics.csv', index=False)

    # return global metrics
    snr_mean, snr_std = compute_metric_avg_std(snr_lst)
    macc_mean, macc_std = compute_metric_avg_std(macc_lst)
    mae_mean, mae_std = compute_mae(hr_pred_lst, hr_label_lst)
    rmse_mean, rmse_std = compute_rmse(hr_pred_lst, hr_label_lst)
    mape_mean, mape_std = compute_mape(hr_pred_lst, hr_label_lst)
    pearson_mean, pearson_std = compute_pearson(hr_pred_lst, hr_label_lst)
    global_metrics = {
        'dataset_name': dataset_name, 'method_name': method_name,
        'snr_mean': snr_mean, 'snr_std': snr_std,
        'macc_mean': macc_mean, 'macc_std': macc_std,
        'mae_mean': mae_mean, 'mae_std': mae_std,
        'rmse_mean': rmse_mean, 'rmse_std': rmse_std,
        'mape_mean': mape_mean, 'mape_std': mape_std,
        'pearson_mean': pearson_mean, 'pearson_std': pearson_std
    }
    print(f'Dataset: {dataset_name}\nMethod: {method_name}\nGlobal metrics: {global_metrics}')
    return global_metrics

def frames2rPPG_unsupervised(frames, fps, method_name):
    """
    video frames to rPPG signal.
    Args:
        frames: video frames,shape (chunk_length, w, h, 3)
    Returns:
        rPPG: rPPG estimation (chunk_length, )
    """
    # unsupervised rppg methods
    if method_name == "POS":
        rppg = POS_WANG(frames, fps)
    elif method_name == "CHROM":
        rppg = CHROME_DEHAAN(frames, fps)
    elif method_name == "ICA":
        rppg = ICA_POH(frames, fps)
    elif method_name == "GREEN":
        rppg = GREEN(frames)
    elif method_name == "LGI":
        rppg = LGI(frames)
    elif method_name == "PBV":
        rppg = PBV(frames)
    elif method_name == "OMIT":
        rppg = OMIT(frames)
    else:
        raise ValueError("unsupervised method name wrong!")

    return rppg
