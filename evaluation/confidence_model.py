import pickle
import scipy
import os
import numpy as np
from tqdm import tqdm
from evaluation.visualization import plot_psd_statistics, create_psd_figure, save_figure, plot_hr_hist
from evaluation.post_process import psd_to_hr
from math import isclose

class StatistcsConfidenceModel:
    def __init__(self, config):
        self.config = config
        self.confidence_low_pass = config.CONFIDENCE_MODEL.BANDPASS_LOW_FREQ
        self.confidence_high_pass = config.CONFIDENCE_MODEL.BANDPASS_HIGH_FREQ
        self.confidence_freq_resolution = config.CONFIDENCE_MODEL.FREQ_RESOLUTION
        self.inference_low_pass = config.POST_PROCESSING.BANDPASS_LOW_FREQ
        self.inference_high_pass = config.POST_PROCESSING.BANDPASS_HIGH_FREQ
        self.inference_freq_resolution = config.POST_PROCESSING.FREQ_RESOLUTION
        self.hr_values = np.arange(int(self.inference_low_pass), int(self.inference_high_pass+self.inference_freq_resolution), int(self.inference_freq_resolution))
        self.confidence_psd_freqs = np.arange(int(self.confidence_low_pass), int(self.confidence_high_pass+self.confidence_freq_resolution), int(self.confidence_freq_resolution))
        self.confidence_psd_len = len(self.confidence_psd_freqs)
        self.psd_statistics = None
        self.psd_statistics_smoothed = None

    def save(self, save_path):
        """
        Save the learned confidence model
        Args:
            save_path: str, path to save the confidence model
        """
        os.makedirs(save_path, exist_ok=True)
        np.savez(os.path.join(save_path, 'psd_statistics.npz'), **self.psd_statistics)
        np.savez(os.path.join(save_path, 'psd_statistics_smoothed.npz'), **self.psd_statistics_smoothed)

    def load(self, load_path):
        """
        Load the learned confidence model
        Args:
            load_path: str, path to load the confidence model
        """
        self.psd_statistics = np.load(os.path.join(load_path, 'psd_statistics.npz'))
        self.psd_statistics_smoothed = np.load(os.path.join(load_path, 'psd_statistics_smoothed.npz'))

    def save_statistics_plots(self, save_path, plot_smoothed=True):
        """
        Save the plots of the psd statistics
        Args:
            save_path: str, path to save the plots
        """
        if plot_smoothed:
            plot_psd_stats = self.psd_statistics_smoothed
            print("saving smoothed stats plots")
        else:
            plot_psd_stats = self.psd_statistics
            print("saving raw stats plots")

        if plot_psd_stats is None:
            raise ValueError("Confidence model not learned or loaded yet")
        
        os.makedirs(save_path, exist_ok=True)
        for hr_idx, hr in tqdm(enumerate(self.hr_values), total=len(self.hr_values)):
            reference_psd = plot_psd_stats['psd_means'][hr_idx]
            reference_psd_std = plot_psd_stats['psd_stds'][hr_idx]
            sample_count = plot_psd_stats['sample_counts'][hr_idx]

            create_psd_figure(title=f'HR: {hr} bpm, sample count: {sample_count:.0f}', xlim=[self.confidence_low_pass, self.confidence_high_pass], xticks=self.confidence_psd_freqs[::2])
            plot_psd_statistics(self.confidence_psd_freqs, reference_psd, reference_psd_std, label='reference')
            save_figure(f'{save_path}/refpsd_{hr}.png')

    def fit(self, psd_stats, hr_stats):
        """
        Fit the confidence model using the ppg_stats and hr_stats
        Args:
            psd_stats: numpy array of shape (n_samples, n_features)
            hr_stats: numpy array of shape (n_samples, )
        """

        # calculate basic statistics for each hr
        hr_classes_num = len(self.hr_values)
        self.psd_statistics = {
            "psd_means": np.zeros((hr_classes_num, self.confidence_psd_len), dtype=np.float32),
            "psd_stds": np.zeros((hr_classes_num, self.confidence_psd_len), dtype=np.float32),
            "sample_counts": np.zeros((hr_classes_num, ), dtype=np.int32)
        }
        for hr_idx, hr in tqdm(enumerate(self.hr_values), total=len(self.hr_values)):
            hr_class_indices = np.where(hr_stats == hr)[0]
            sample_count = len(hr_class_indices)

            if sample_count == 0:
                reference_psd = np.zeros(self.confidence_psd_len)
                reference_psd_std = np.zeros(self.confidence_psd_len)
            else:
                reference_psd = np.mean(psd_stats[hr_class_indices], axis=0)
                reference_psd_std = np.std(psd_stats[hr_class_indices], axis=0)

            self.psd_statistics["psd_means"][hr_idx, :] = reference_psd
            self.psd_statistics["psd_stds"][hr_idx, :] = reference_psd_std
            self.psd_statistics["sample_counts"][hr_idx] = sample_count

        # weighted smoothing to 1. address data imbalance 2. introduce continuity
        print("weighted smoothing...")
        smoothing_window_size = 5
        self.psd_statistics_smoothed = {
            "psd_means": np.zeros((hr_classes_num, self.confidence_psd_len), dtype=np.float32),
            "psd_stds": np.zeros((hr_classes_num, self.confidence_psd_len), dtype=np.float32),
            "sample_counts": np.zeros((hr_classes_num, ), dtype=np.int32)
        }
        for hr_idx, hr in tqdm(enumerate(self.hr_values), total=len(self.hr_values)):
            window_start, window_end = max(0, hr_idx - smoothing_window_size // 2), min(hr_classes_num-1, hr_idx + smoothing_window_size // 2) + 1
            
            # adjust psd means to match current hr
            window_size = window_end - window_start
            window_psd_means = np.zeros((window_size, self.confidence_psd_len))
            window_psd_stds = np.zeros((window_size, self.confidence_psd_len))
            for window_hr_idx in range(window_start, window_end):
                current_psd_mean = self.psd_statistics["psd_means"][window_hr_idx]
                current_psd_std = self.psd_statistics["psd_stds"][window_hr_idx]

                scaling_factor = hr / self.hr_values[window_hr_idx]
                psd_mean_scaled = np.interp(self.confidence_psd_freqs, self.confidence_psd_freqs * scaling_factor, current_psd_mean, left=0, right=0)
                psd_std_scaled = np.interp(self.confidence_psd_freqs, self.confidence_psd_freqs * scaling_factor, current_psd_std, left=0, right=0)

                window_psd_means[window_hr_idx - window_start, :] = psd_mean_scaled
                window_psd_stds[window_hr_idx - window_start, :] = psd_std_scaled

            # weighted average in window
            window_sample_counts = self.psd_statistics["sample_counts"][window_start:window_end]
            if np.sum(window_sample_counts) == 0: # no samples in window
                self.psd_statistics_smoothed["psd_means"][hr_idx, :] = np.zeros(self.confidence_psd_len)
                self.psd_statistics_smoothed["psd_stds"][hr_idx, :] = np.zeros(self.confidence_psd_len)
                self.psd_statistics_smoothed["sample_counts"][hr_idx] = 0
            else:
                window_weights = window_sample_counts / np.sum(window_sample_counts)
                weighted_psd_mean = np.dot(window_weights, window_psd_means)
                weighted_psd_std = np.dot(window_weights, window_psd_stds)
                self.psd_statistics_smoothed["psd_means"][hr_idx, :] = weighted_psd_mean
                self.psd_statistics_smoothed["psd_stds"][hr_idx, :] = weighted_psd_std
                self.psd_statistics_smoothed["sample_counts"][hr_idx] = np.sum(window_sample_counts)

    def predict(self, hr_pred, freq_rppg, psd_rppg, confidence_type="distance"):
        """
        Predict the confidence given predicted rppg psd
        Args:
            hr_pred: float, predicted hr
            freq_rppg: numpy array of shape (n_freqs, ), frequency of the rppg psd
            psd_rppg: numpy array of shape (n_freqs, ), power of the rppg psd
            confidence_type: str, type of confidence to calculate
        """
        if self.psd_statistics is None: # check if confidence model is learned or loaded
            raise ValueError("Confidence model not learned or loaded yet")
        if psd_rppg.shape != (self.confidence_psd_len, ): # check shape
            raise ValueError(f"PSD shape mismatch, psd_rppg: {psd_rppg.shape}, reference_psd: {(self.confidence_psd_len, )}")
        if hr_pred < self.inference_low_pass or hr_pred > self.inference_high_pass: # check if hr is in the range
            raise ValueError(f"Predicted HR {hr_pred} is out of range {self.inference_low_pass} - {self.inference_high_pass} bpm")
        if not np.array_equal(freq_rppg, self.confidence_psd_freqs): # check if freqs match
            raise ValueError(f"PSD frequency mismatch, freq_rppg: {freq_rppg}, reference_freqs: {self.confidence_psd_freqs}")

        # get statistics of the predicted psd
        rppg_hr_idx = np.argmin(np.abs(hr_pred - self.hr_values))
        reference_psd = self.psd_statistics["psd_means"][rppg_hr_idx]
        reference_psd_std = self.psd_statistics["psd_stds"][rppg_hr_idx]

        # check if both psds are normalized
        assert isclose(np.sum(psd_rppg), 1, abs_tol=1e-3), f"{np.sum(psd_rppg)=} not normalized"
        assert isclose(np.sum(reference_psd), 1, abs_tol=1e-3), f"{np.sum(reference_psd)=} not normalized"

        # calculate confidence score
        if confidence_type == "probability": # probability of being in the reference psd distribution
            # TODO: how to calculate the joint prob???
            # calculate z score
            epsilon = 1e-6
            std_with_epsilon = reference_psd_std + epsilon
            z_score = np.abs((psd_rppg - reference_psd) / std_with_epsilon)
            # calculate confidence under gaussians prior
            cdf = scipy.stats.norm.cdf(z_score)
            confidence = 1 - 2 * np.abs(0.5 - cdf)
        elif confidence_type == "distance": # total variance distance to the reference psd
            tvd = 0.5 * np.sum(np.abs(psd_rppg - reference_psd))
            confidence = 1 - tvd
        else:
            raise ValueError(f"Invalid confidence type: {confidence_type}")

        return confidence
    