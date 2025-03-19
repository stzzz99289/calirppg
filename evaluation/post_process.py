"""The post processing files for caluclating heart rate using FFT or peak detection.
The file also  includes helper funcs such as detrend, power2db etc.
"""

import numpy as np
import scipy
import scipy.io
from math import ceil
from scipy.signal import butter
from scipy.sparse import spdiags
from copy import deepcopy
from evaluation.utils import _next_power_of_2

def _detrend(input_signal, lambda_value):
    """Detrend PPG signal."""
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    detrended_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return detrended_signal

def post_processing(rPPG, fs, diff_flag=False, use_bandpass=True, low_pass=45, high_pass=150):
    # if the predictions and labels are 1st derivative of PPG signal.
    if diff_flag:  
        post_processed_rPPG = _detrend(np.cumsum(rPPG), 100)
    else:
        post_processed_rPPG = _detrend(rPPG, 100)

    # bandpass filter between [low_pass, high_pass] bpm
    if use_bandpass:
        [b, a] = butter(1, [low_pass / 60 / fs * 2, high_pass / 60 / fs * 2], btype='bandpass')
        post_processed_rPPG = scipy.signal.filtfilt(b, a, np.double(post_processed_rPPG))
    return post_processed_rPPG

def signal_to_psd(signal, fs, freq_resolution=None, low_pass=45, high_pass=150, normalization=True, interpolation=False):
    """Convert rPPG/PPG signal to PSD."""
    signal_length = signal.shape[0]

    # (fs / freq_resolution) is nfft value that can achieve the desired freq resolution
    # rppg_length is the lower bound of nfft value to include all spectral information
    if freq_resolution is None:
        nfft = signal_length
    else:
        nfft = max(ceil(fs / freq_resolution), signal_length)

    # use power of 2 as nfft value for computational efficiency
    nfft = _next_power_of_2(nfft)

    # calculate psd
    freq, psd = scipy.signal.periodogram(signal, fs=fs, nfft=nfft, detrend=False)
    bandpass_mask = np.argwhere((freq >= low_pass / 60) & (freq <= high_pass / 60))
    freq = np.take(freq, bandpass_mask) * 60 # transform freq unit to bpm
    psd = np.take(psd, bandpass_mask)

    # interpolation to get constant integer freq bpm values
    if (freq_resolution is not None) and interpolation:
        freq_int = np.arange(int(low_pass), int(high_pass+freq_resolution), int(freq_resolution))
        psd = np.interp(freq_int.reshape(-1), freq.reshape(-1), psd.reshape(-1))
        freq = freq_int

    # normalization to make area under curve equal to 1
    if normalization:
        psd = psd / np.sum(psd)

    return freq, psd

def psd_to_hr(freq, psd):
    """Convert PSD to HR in bpm."""
    # find the max psd index
    fft_hr = np.take(freq, np.argmax(psd, 0))
    return fft_hr

