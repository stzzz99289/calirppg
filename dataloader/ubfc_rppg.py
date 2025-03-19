"""The dataloader for UBFC-rPPG dataset.

Details for the UBFC-rPPG Dataset see https://sites.google.com/view/ybenezeth/ubfcrppg.
If you use this dataset, please cite this paper:
S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois, "Unsupervised skin tissue segmentation for remote photoplethysmography", Pattern Recognition Letters, 2017.
"""
import glob
import os
import re
import cv2
import numpy as np
from dataloader.base import BaseLoader


class UBFCrPPGLoader(BaseLoader):
    """The data loader for the UBFC-rPPG dataset."""

    def __init__(self, config_data):
        """Initializes an UBFC-rPPG dataloader.
            data_path should be "RawData" for below dataset structure:
            -----------------
                RawData/
                |   |-- subject1/
                |       |-- vid.avi
                |       |-- ground_truth.txt
                |   |-- subject2/
                |       |-- vid.avi
                |       |-- ground_truth.txt
                |...
                |   |-- subjectn/
                |       |-- vid.avi
                |       |-- ground_truth.txt
            -----------------
        """
        super().__init__(config_data)

    def get_raw_data_dirs(self, data_path):
        data_dirs = glob.glob(data_path + os.sep + "subject*")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        
        dirs = list()
        for data_dir in data_dirs:
            subject_name = re.search('subject(\d+)', data_dir).group(0)
            dirs.append({"path": data_dir, "subject": subject_name, "index": None})
        dirs.sort(key=lambda x: (x['subject'], x['index']))
        
        return dirs
    
    def video_frame_generator(self, data_dir):
        data_path = data_dir['path']
        subject_name = data_dir['subject']
        trail_index = data_dir['index']
        video_path = data_path

        VidObj = cv2.VideoCapture(os.path.join(video_path, "vid.avi"))
        if not VidObj.isOpened():
            raise ValueError(f"wrong video path: {video_path}")
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        frame_idx = 0
        while True:
            success, frame = VidObj.read()
            if not success:
                break # end of video
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            yield frame_idx, frame
            frame_idx += 1
        VidObj.release()

    def read_raw_bvp_wave(self, data_dir):
        data_path = data_dir['path']
        subject_name = data_dir['subject']
        trail_index = data_dir['index']
        bvp_file = os.path.join(data_path, f'ground_truth.txt')
        bvp = []
        with open(bvp_file, "r") as f:
            str1 = f.read()
            str1 = str1.split("\n")
            bvp = [float(x) for x in str1[0].split()]
            waves = np.asarray(bvp)
            sq_vec = np.ones_like(waves)
        return waves, sq_vec