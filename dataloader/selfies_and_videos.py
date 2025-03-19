"""The dataloader for Selfies-and-Videos datasets.

Details for the Selfies-and-Videos Dataset see https://www.kaggle.com/datasets/tapakah68/selfies-and-video-dataset-4-000-people?resource=download
"""
import glob
import os
import cv2
import numpy as np
from dataloader.base import BaseLoader


class SelfiesAndVideosLoader(BaseLoader):
    """The data loader for the Selfies-and-Videos dataset."""

    def __init__(self, config_data):
        """Initializes an Selfies-and-Videos dataloader.
            THIS IS A REFERENCE-ONLY DATASET WITH NO GROUND TRUTH PPG LABELS.
            data_path should be "RawData" for below dataset structure:
            -----------------
                RawData/
                |   |-- 1
                |      |-- 3.mp4
                |      |-- 4.mp4
                |      |-- ...
                |   |-- 2
                |      |-- 3.mp4
                |      |-- 4.mp4
                |      |-- ...
                |   |-- ...
            -----------------
        """
        super().__init__(config_data)

    def get_raw_data_dirs(self, data_path):
        data_dirs = glob.glob(data_path + os.sep + "*" + os.sep + "*.mp4")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        
        dirs = list()
        for data_dir in data_dirs:
            subject = os.path.basename(os.path.dirname(data_dir))
            index = os.path.basename(data_dir).split('.')[0]
            dirs.append({"path": data_dir, "subject": subject, "index": index})
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
        # reference-only dataset, no ground truth PPG labels, use all zeros as pseudo PPG labels
        waves = np.zeros(10)
        sq_vec = np.ones_like(waves)
        return waves, sq_vec