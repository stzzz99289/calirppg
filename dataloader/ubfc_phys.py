"""The dataloader for the UBFC-PHYS dataset.

Details for the UBFC-PHYS Dataset see https://sites.google.com/view/ybenezeth/ubfc-phys.
If you use this dataset, please cite this paper:
R. Meziati Sabour, Y. Benezeth, P. De Oliveira, J. Chappé, F. Yang. 
"UBFC-Phys: A Multimodal Database For Psychophysiological Studies Of Social Stress", 
IEEE Transactions on Affective Computing, 2021.
"""
import glob
import os
import re
import cv2
import csv
import numpy as np
from dataloader.base import BaseLoader

class UBFCPhysLoader(BaseLoader):
    """The data loader for the UBFC-PHYS dataset."""

    def __init__(self, config_data):
        """Initializes an UBFC-PHYS dataloader.
            data_path should be "RawData" for below dataset structure:
            -----------------
                RawData/
                |   |-- s1/
                |       |-- vid_s1_T1.avi
                |       |-- vid_s1_T2.avi
                |       |-- vid_s1_T3.avi
                |       |...
                |       |-- bvp_s1_T1.csv
                |       |-- bvp_s1_T2.csv
                |       |-- bvp_s1_T3.csv
                |   |-- s2/
                |       |-- vid_s2_T1.avi
                |       |-- vid_s2_T2.avi
                |       |-- vid_s2_T3.avi
                |       |...
                |       |-- bvp_s2_T1.csv
                |       |-- bvp_s2_T2.csv
                |       |-- bvp_s2_T3.csv
                |...
                |   |-- sn/
                |       |-- vid_sn_T1.avi
                |       |-- vid_sn_T2.avi
                |       |-- vid_sn_T3.avi
                |       |...
                |       |-- bvp_sn_T1.csv
                |       |-- bvp_sn_T2.csv
                |       |-- bvp_sn_T3.csv
            -----------------
        """
        super().__init__(config_data)

    def get_raw_data_dirs(self, data_path):
        data_dirs = glob.glob(data_path + os.sep + "s*" + os.sep + "vid_*.avi")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        
        dirs = list()
        for data_dir in data_dirs:
            video_name = re.search('vid_(.*).avi', data_dir).group(1)
            subject_name = video_name.split('_')[0]
            trail_index = video_name.split('_')[1]
            dirs.append({"path": data_dir, "subject": subject_name, "index": trail_index})
        dirs.sort(key=lambda x: (x['subject'], x['index']))

        return dirs
    
    def video_frame_generator(self, data_dir):
        data_path = data_dir['path']
        subject_name = data_dir['subject']
        trail_index = data_dir['index']
        video_path = data_path
        
        VidObj = cv2.VideoCapture(video_path)
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
        bvp_file = data_path.replace("vid", "bvp").replace(".avi", ".csv")
        bvp = []
        with open(bvp_file, "r") as f:
            d = csv.reader(f)
            for row in d:
                bvp.append(float(row[0]))
            waves = np.asarray(bvp)
            sq_vec = np.ones_like(waves)
        return waves, sq_vec