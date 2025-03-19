"""The dataloader for iBVP datasets.

Details for the iBVP Dataset see https://doi.org/10.3390/electronics13071334
If you use this dataset, please cite the following publications:

Joshi, Jitesh, and Youngjun Cho. 2024. "iBVP Dataset: RGB-Thermal rPPG Dataset with High Resolution Signal Quality Labels" Electronics 13, no. 7: 1334. https://doi.org/10.3390/electronics13071334 

Joshi, Jitesh, Katherine Wang, and Youngjun Cho. 2023. "PhysioKit: An Open-Source, Low-Cost Physiological Computing Toolkit for Single- and Multi-User Studies" Sensors 23, no. 19: 8244. https://doi.org/10.3390/s23198244 

"""
import glob
import os
import cv2
import pandas as pd
from dataloader.base import BaseLoader

class iBVPLoader(BaseLoader):
    """The data loader for the iBVP dataset."""

    def __init__(self, config_data):
        """Initializes an iBVP dataloader.
            data_path should be "iBVP_Dataset" for below dataset structure:
            -----------------
                iBVP_Dataset/
                |   |-- p01_a/
                |      |-- p01_a_rgb/
                |      |-- p01_a_t/
                |      |-- p01_a_bvp.csv
                |   |-- p01_b/
                |      |-- p01_b_rgb/
                |      |-- p01_b_t/
                |      |-- p01_b_bvp.csv
                |...
                |   |-- pii_x/
                |      |-- pii_x_rgb/
                |      |-- pii_x_t/
                |      |-- pii_x_bvp.csv
            -----------------
        """
        super().__init__(config_data)

    def get_raw_data_dirs(self, data_path):
        data_dirs = glob.glob(data_path + os.sep + "*_*")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        
        dirs = list()
        for data_dir in data_dirs:
            subject_trail_val = os.path.split(data_dir)[-1].replace('_', '')
            index = subject_trail_val[3]
            subject = subject_trail_val[0:3]
            dirs.append({"path": data_dir, "subject": subject, "index": index})
        dirs.sort(key=lambda x: (x['subject'], x['index']))

        return dirs
    
    def video_frame_generator(self, data_dir):
        data_path = data_dir['path']
        subject_name = data_dir['subject']
        trail_index = data_dir['index']
        search_bmp_path = os.path.join(data_path, f'{subject_name}_{trail_index}_rgb', '*.bmp')
        all_bmp = sorted(glob.glob(search_bmp_path))
        for frame_idx, bmp_path in enumerate(all_bmp):
            img = cv2.imread(bmp_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            yield frame_idx, img

    def read_raw_bvp_wave(self, data_dir):
        data_path = data_dir['path']
        subject_name = data_dir['subject']
        trail_index = data_dir['index']
        bvp_file = os.path.join(data_path, f'{subject_name}_{trail_index}_bvp.csv')
        with open(bvp_file, "r") as f:
            labels = pd.read_csv(f).to_numpy()
            waves = labels[:, 0]
            sq_vec = labels[:, 2]
        return waves, sq_vec