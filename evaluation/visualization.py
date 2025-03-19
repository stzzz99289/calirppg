import matplotlib.pyplot as plt
import numpy as np

def create_psd_figure(title="psd plot", xlim=[45, 150], ylim=[0, 0.30], xticks=np.arange(45, 151, 2)):
    xlabel = 'HR (bpm)'
    ylabel = 'power'
    
    plt.figure(figsize=(20, 5))
    plt.title(title)
    plt.xlim(xlim); plt.ylim(ylim)
    plt.xticks(xticks)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout()
    for i in range(len(xticks)):
        plt.axvline(xticks[i], linestyle='--', color='black', alpha=0.3)

def plot_psd(freq, psd, label=None, alpha=1):
    plt.plot(freq, psd, 'o--', markersize=5, label=label, alpha=alpha)

def plot_psd_statistics(freq, psd_ref, psd_std, label='reference', alpha=1.0):
    """Plot PSD statistics with mean and shaded standard deviation region
    Args:
        freq: frequency values (x-axis)
        psd_ref: reference/mean PSD values (y-axis)
        psd_std: standard deviation of PSD values
        color: color for the plot
        label: label for the legend
    """
    # Plot mean line
    plt.plot(freq, psd_ref, 'o--', markersize=5, color='red', label=label, alpha=alpha)
    
    # Plot shaded region for standard deviation
    plt.fill_between(freq, 
                    psd_ref - psd_std, 
                    psd_ref + psd_std, 
                    color='red', 
                    alpha=alpha*0.5)

def plot_hr_hist(hr_stats, title="Histogram of HR", min_hr=45, max_hr=150, step=2):
    plt.figure(figsize=(10, 5))
    bins = np.arange(min_hr, max_hr + step, step)
    counts, bins, patches = plt.hist(hr_stats, bins=bins)
    for patch, count in zip(patches, counts):
        height = patch.get_height()
        plt.annotate(f'{count:.0f}', xy=(patch.get_x() + patch.get_width() / 2, height),
                     xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')
    plt.xticks(np.arange(min_hr, max_hr + step, step*2))
    plt.xlabel('HR (bpm)')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.tight_layout()

def save_figure(save_path):
    plt.legend()
    plt.savefig(save_path)
    plt.close()