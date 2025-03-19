import numpy as np
import scipy
import scipy.io
from evaluation.utils import _next_power_of_2, power2db
from copy import deepcopy

def compute_macc(pred_signal, gt_signal):
    """Calculate maximum amplitude of cross correlation (MACC) by computing correlation at all time lags.
        Args:
            pred_ppg_signal(np.array): predicted PPG signal 
            label_ppg_signal(np.array): ground truth, label PPG signal
        Returns:
            MACC(float): Maximum Amplitude of Cross-Correlation
    """
    pred = deepcopy(pred_signal)
    gt = deepcopy(gt_signal)
    pred = np.squeeze(pred)
    gt = np.squeeze(gt)
    min_len = np.min((len(pred), len(gt)))
    pred = pred[:min_len]
    gt = gt[:min_len]
    lags = np.arange(0, len(pred)-1, 1)
    tlcc_list = []
    for lag in lags:
        cross_corr = np.abs(np.corrcoef(
            pred, np.roll(gt, lag))[0][1])
        tlcc_list.append(cross_corr)
    macc = max(tlcc_list)
    return macc

def compute_snr(pred_ppg_signal, hr_label, fs=30, low_pass=0.75, high_pass=2.5):
    """Calculate SNR as the ratio of the area under the curve of the frequency spectrum around the first and second harmonics 
        of the ground truth HR frequency to the area under the curve of the remainder of the frequency spectrum, from 0.75 Hz
        to 2.5 Hz. 

        Args:
            pred_ppg_signal(np.array): predicted PPG signal 
            label_ppg_signal(np.array): ground truth, label PPG signal
            fs(int or float): sampling rate of the video
        Returns:
            SNR(float): Signal-to-Noise Ratio
    """
    # Get the first and second harmonics of the ground truth HR in Hz
    first_harmonic_freq = hr_label / 60
    second_harmonic_freq = 2 * first_harmonic_freq
    deviation = 6 / 60  # 6 beats/min converted to Hz (1 Hz = 60 beats/min)

    # Calculate FFT
    pred_ppg_signal = np.expand_dims(pred_ppg_signal, 0)
    N = _next_power_of_2(pred_ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(pred_ppg_signal, fs=fs, nfft=N, detrend=False)

    # Calculate the indices corresponding to the frequency ranges
    idx_harmonic1 = np.argwhere((f_ppg >= (first_harmonic_freq - deviation)) & (f_ppg <= (first_harmonic_freq + deviation)))
    idx_harmonic2 = np.argwhere((f_ppg >= (second_harmonic_freq - deviation)) & (f_ppg <= (second_harmonic_freq + deviation)))
    idx_remainder = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass) \
     & ~((f_ppg >= (first_harmonic_freq - deviation)) & (f_ppg <= (first_harmonic_freq + deviation))) \
     & ~((f_ppg >= (second_harmonic_freq - deviation)) & (f_ppg <= (second_harmonic_freq + deviation))))

    # Select the corresponding values from the periodogram
    pxx_ppg = np.squeeze(pxx_ppg)
    pxx_harmonic1 = pxx_ppg[idx_harmonic1]
    pxx_harmonic2 = pxx_ppg[idx_harmonic2]
    pxx_remainder = pxx_ppg[idx_remainder]

    # Calculate the signal power
    signal_power_hm1 = np.sum(pxx_harmonic1)
    signal_power_hm2 = np.sum(pxx_harmonic2)
    signal_power_rem = np.sum(pxx_remainder)

    # Calculate the SNR as the ratio of the areas
    if not signal_power_rem == 0: # catches divide by 0 runtime warning 
        SNR = power2db((signal_power_hm1 + signal_power_hm2) / signal_power_rem)
    else:
        SNR = 0
    return SNR

def compute_mae(hr_pred_lst, hr_label_lst):
    """Calculate mean absolute error (MAE) between predicted and label HRs."""
    mae = np.mean(np.abs(hr_pred_lst - hr_label_lst))
    standard_error = np.std(np.abs(hr_pred_lst - hr_label_lst)) / np.sqrt(len(hr_pred_lst))
    return mae, standard_error

def compute_rmse(hr_pred_lst, hr_label_lst):
    """Calculate root mean square error (RMSE) between predicted and label HRs."""
    squared_errors = (hr_pred_lst - hr_label_lst) ** 2
    rmse = np.sqrt(np.mean(squared_errors))
    standard_error = np.sqrt(np.std(squared_errors) / np.sqrt(len(hr_pred_lst)))
    return rmse, standard_error

def compute_mape(hr_pred_lst, hr_label_lst):
    """Calculate mean absolute percentage error (MAPE) between predicted and label HRs."""
    mape = np.mean(np.abs((hr_pred_lst - hr_label_lst) / hr_label_lst)) * 100
    standard_error = np.std(np.abs((hr_pred_lst - hr_label_lst) / hr_label_lst)) / np.sqrt(len(hr_pred_lst)) * 100
    return mape, standard_error

def compute_pearson(hr_pred_lst, hr_label_lst):
    """Calculate Pearson correlation coefficient between predicted and label HRs."""
    pearson = np.corrcoef(hr_pred_lst, hr_label_lst)[0][1]
    standard_error = np.sqrt((1 - pearson**2) / (len(hr_pred_lst) - 2))
    return pearson, standard_error

def compute_metric_avg_std(metric_lst):
    """Calculate average and standard deviation of a list of metrics."""
    avg = np.mean(metric_lst)
    std = np.std(metric_lst) / np.sqrt(len(metric_lst))
    return avg, std