import pickle
import scipy
import os
import numpy as np
from tqdm import tqdm
from evaluation.visualization import plot_psd_statistics, create_psd_figure, save_figure, plot_hr_hist
from evaluation.post_process import psd_to_hr
from math import isclose
from scipy import stats
from matplotlib import pyplot as plt

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
        self.psd_m_dists = None

    def save(self, save_path):
        """
        Save the learned confidence model
        Args:
            save_path: str, path to save the confidence model
        """
        os.makedirs(save_path, exist_ok=True)
        np.savez(os.path.join(save_path, 'psd_statistics.npz'), **self.psd_statistics)
        np.savez(os.path.join(save_path, 'psd_m_dists.npz'), **self.psd_m_dists)

    def load(self, load_path):
        """
        Load the learned confidence model
        Args:
            load_path: str, path to load the confidence model
        """
        self.psd_statistics = np.load(os.path.join(load_path, 'psd_statistics.npz'))
        self.psd_m_dists = np.load(os.path.join(load_path, 'psd_m_dists.npz'))

    def save_statistics_plots(self, save_path):
        """
        Save the plots of the psd statistics
        Args:
            save_path: str, path to save the plots
        """
        plot_psd_stats = self.psd_statistics
        if plot_psd_stats is None:
            raise ValueError("Confidence model not learned or loaded yet")
        
        os.makedirs(save_path, exist_ok=True)
        for hr_idx, hr in tqdm(enumerate(self.hr_values), total=len(self.hr_values)):
            psd_mean = plot_psd_stats['psd_means'][hr_idx]
            psd_covariance = plot_psd_stats['psd_covariances'][hr_idx]
            sample_count = plot_psd_stats['sample_counts'][hr_idx]

            create_psd_figure(title=f'condition HR = {hr} bpm', xlim=[self.confidence_low_pass, self.confidence_high_pass], xticks=self.confidence_psd_freqs[::5])
            plot_psd_statistics(self.confidence_psd_freqs, psd_mean, psd_covariance)
            plt.savefig(f'{save_path}/psd_distribution_hr={hr:03d}.svg', dpi=300, bbox_inches='tight', format='svg')
            plt.close()
            # save_figure(f'{save_path}/psd_distribution_hr={hr:03d}.png')

    def psds_scaling(self, psds, psd_hr, target_hr):
        """
        Scale psds to the target hr
        Args:
            psds: numpy array of shape (n_samples, psd_len)
            psd_hr: float, hr of the psd
            target_hr: float, target hr
        Returns:
            numpy array of shape (n_samples, psd_len)
        """
        scaling_factor = target_hr / psd_hr
        return np.array([np.interp(self.confidence_psd_freqs, self.confidence_psd_freqs * scaling_factor, psd, left=0, right=0) for psd in psds])

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
            "psd_means": np.zeros((hr_classes_num, self.confidence_psd_len), dtype=np.float32), # psd mean of multivariate gaussian
            "psd_covariances": np.zeros((hr_classes_num, self.confidence_psd_len, self.confidence_psd_len), dtype=np.float32), # psd covariance of multivariate gaussian
            "sample_counts": np.zeros((hr_classes_num, ), dtype=np.int32),
        }
        self.psd_m_dists = {}
        window_extent = 4 # window extent from center length (window_length=2*window_extent+1)
        for hr_idx, hr in tqdm(enumerate(self.hr_values), total=len(self.hr_values)):
            # psd observations in the window of [window_start, window_end)
            window_start, window_end = max(0, hr_idx - window_extent), min(hr_classes_num-1, hr_idx + window_extent) + 1
            window_size = window_end - window_start

            # 1. distance to center discountings
            distances_to_center = np.abs(np.arange(window_start, window_end) - hr_idx)
            distance_weights = np.exp(-0.5 * distances_to_center)

            # 2. sample count weighting
            sample_count_window = np.zeros((window_size, ), dtype=np.int32)
            for window_hr_idx in range(window_start, window_end):
                sample_count_window[window_hr_idx - window_start] = np.sum(hr_stats == self.hr_values[window_hr_idx])
            density_weights = sample_count_window / np.sum(sample_count_window)

            # 3. combine distance and sample count weights
            weights = distance_weights * density_weights
            weights = weights / np.sum(weights) # normalize to make sum=1

            # 4. collect psd observations in hr window with corresponding weights
            psd_observations_window = []
            weight_observations_window = []
            for window_hr_idx in range(window_start, window_end):
                current_hr = self.hr_values[window_hr_idx]
                hr_indices = np.where(hr_stats == current_hr)[0]
                if len(hr_indices) == 0:
                    continue
                psd_observations = psd_stats[hr_indices]
                psd_observations = self.psds_scaling(psd_observations, psd_hr=current_hr, target_hr=hr)
                weight_observations = weights[window_hr_idx - window_start] * np.ones(len(hr_indices))
                psd_observations_window.append(psd_observations)
                weight_observations_window.append(weight_observations)
            psd_observations_window = np.concatenate(psd_observations_window, axis=0)
            weight_observations_window = np.concatenate(weight_observations_window, axis=0)

            # 5. calculate weighted psd mean and covariance
            sample_count = np.sum(sample_count_window)
            epsilon = 1e-6 # Add small constant to diagonal to ensure positive definiteness of cov matrix
            if sample_count == 0:
                # no psd samples in window, set mean and covariance to 0
                psd_mean = np.zeros(self.confidence_psd_len)
                psd_covariance = np.zeros((self.confidence_psd_len, self.confidence_psd_len))
                psd_covariance = psd_covariance + epsilon * np.eye(self.confidence_psd_len)
            else:
                psd_mean = np.average(psd_observations_window, axis=0, weights=weight_observations_window)
                psd_mean = psd_mean / np.sum(psd_mean)
                psd_covariance = np.cov(psd_observations_window, rowvar=False, aweights=weight_observations_window)
                psd_covariance = psd_covariance / (np.sum(psd_mean) ** 2)
                psd_covariance = psd_covariance + epsilon * np.eye(self.confidence_psd_len)

            self.psd_statistics["psd_means"][hr_idx, :] = psd_mean
            self.psd_statistics["psd_covariances"][hr_idx, :, :] = psd_covariance
            self.psd_statistics["sample_counts"][hr_idx] = sample_count
            # calculate mahalanobis distances of samples
            self.psd_m_dists[f"{hr:03d}"] = StatistcsConfidenceModel.mahalanobis_distance(psd_observations_window, psd_mean, psd_covariance)

    def predict(self, hr_pred, freq_rppg, psd_rppg, confidence_type="pvalue"):
        """
        Predict the confidence given predicted rppg psd
        Args:
            hr_pred: float, predicted hr
            freq_rppg: numpy array of shape (n_freqs, ), frequency of the rppg psd
            psd_rppg: numpy array of shape (n_freqs, ), power of the rppg psd
            confidence_type: str, type of confidence to calculate
            use_smoothed: bool, whether to use smoothed statistics
        """        
        if self.psd_statistics is None: # check if confidence model is learned or loaded
            raise ValueError("Confidence model not learned or loaded yet")
        if hr_pred < self.hr_values[0] or hr_pred > self.hr_values[-1]: # check if hr is in the range
            raise ValueError(f"Predicted HR {hr_pred} is out of range {self.hr_values[0]} - {self.hr_values[-1]} bpm")
        if not np.array_equal(freq_rppg, self.confidence_psd_freqs): # check if freqs match
            raise ValueError(f"PSD frequency components mismatch, freq_rppg: {freq_rppg}, reference_freqs: {self.confidence_psd_freqs}")

        # get statistics of the predicted psd
        rppg_hr_idx = np.argmin(np.abs(hr_pred - self.hr_values))
        psd_mean = self.psd_statistics["psd_means"][rppg_hr_idx]
        psd_covariance = self.psd_statistics["psd_covariances"][rppg_hr_idx]
        if isclose(np.sum(psd_mean), 0, abs_tol=1e-3):
            raise ValueError(f"Warning: reference psd is missing for hr={hr_pred} bpm")

        # check if both psds are normalized
        assert isclose(np.sum(psd_rppg), 1, abs_tol=1e-3), f"{np.sum(psd_rppg)=} not normalized"
        assert isclose(np.sum(psd_mean), 1, abs_tol=1e-3), f"{np.sum(psd_mean)=} not normalized"

        # calculate confidence score
        if confidence_type == "pvalue": # p-value of multivariate gaussian
            confidence = StatistcsConfidenceModel.mahalanobis_pvalue(psd_rppg, psd_mean, psd_covariance)
        elif confidence_type == "percentile": # percentile of multivariate gaussian
            confidence = StatistcsConfidenceModel.mahalanobis_percentile(psd_rppg, self.psd_m_dists[f"{hr_pred:03d}"], psd_mean, psd_covariance)
        elif confidence_type == "distance": # total variance distance to the mean
            tvd = 0.5 * np.sum(np.abs(psd_rppg - psd_mean))
            confidence = 1 - tvd
        else:
            raise ValueError(f"Invalid confidence type: {confidence_type}")

        return confidence
    
    def confidence_to_std(self, confidence, c=0.01):
        # if choose c=0.01:
        # when z_score=0 (which means confidence=0) and resolution=2 -> hr_std=100 bpm
        # when z_score=inf (which means confidence=1) and resolution=2 -> hr_std=0 bpm
        z_score = stats.norm.ppf((confidence + 1) / 2)
        hr_std = (self.confidence_freq_resolution / 2) / (z_score + c) + 1e-3
        if isclose(hr_std, 0, abs_tol=1e-4):
            raise ValueError(f"hr_std={hr_std} bpm")
        return hr_std
    
    @staticmethod
    def mahalanobis_percentile(x, m_dists, mean, cov):
        """
        Calculate the percentile of the Mahalanobis distance given distance samples
        """
        md_x = StatistcsConfidenceModel.mahalanobis_distance(x, mean, cov)
        return np.mean(m_dists >= md_x)
    
    @staticmethod
    def mahalanobis_distance(x, mean, cov, epsilon=1e-6):
        """
        Calculate the Mahalanobis distance (multivariate z-score equivalent)
        
        Parameters:
        x: The point to evaluate (can be a single point or array of points)
        mean: Mean vector of the distribution
        cov: Covariance matrix of the distribution
        
        Returns:
        The Mahalanobis distance(s)
        """
        x_minus_mu = x - mean
        inv_cov = np.linalg.inv(cov)
        
        # For a single point
        if x.ndim == 1:
            return np.sqrt(x_minus_mu.dot(inv_cov).dot(x_minus_mu))
        
        # For multiple points
        result = np.zeros(len(x))
        for i, x_i in enumerate(x):
            x_i_minus_mu = x_i - mean
            result[i] = np.sqrt(x_i_minus_mu.dot(inv_cov).dot(x_i_minus_mu))
        
        return result
    
    @staticmethod
    def mahalanobis_pvalue(x, mean, cov):
        """
        Calculate p-value for Mahalanobis distance
        (probability of observing a point at least as extreme as x)
        """
        m_dist = StatistcsConfidenceModel.mahalanobis_distance(x, mean, cov)
        scaled_m_dist = m_dist / np.sqrt(len(mean)) # scale the distance by the square root of the number of features to avoid curse of dimensionality
        c = 5 # constant for sigmoid function, when c=5, p_value~=0.99 for m_dist=0.
        p_value = 1 / (1 + np.exp(c * (scaled_m_dist - 1)))
        # p_value = 1 - stats.chi2.cdf(m_dist**2, df=len(mean))
        return p_value