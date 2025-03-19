"""The dataloader for MAHNOB-HCI datasets.

Details for the MAHNOB-HCI Dataset see https://mahnob-db.eu/hci-tagging/
"""
import glob
import os
import cv2
import json
import pyedflib
import numpy as np
from dataloader.base import BaseLoader

class MAHNOBLoader(BaseLoader):
    """The data loader for the MAHNOB-HCI dataset."""

    def __init__(self, config_data):
        """Initializes an MAHNOB-HCI dataloader.
            data_path should be "RawData" for below dataset structure:
            -----------------
                RawData/
                |   |-- 2/
                |      |-- P1-Rec1-2009.07.09.17.53.46_C1 trigger _C_Section_2.avi
                |      |-- Part_1_S_Trial1_emotion.bdf
                |   |-- 4/
                |      |-- P1-Rec1-2009.07.09.17.53.46_C1 trigger _C_Section_4.avi
                |      |-- Part_1_S_Trial2_emotion.bdf
                |...
            -----------------
        """
        super().__init__(config_data)

    def get_raw_data_dirs(self, data_path):
        data_dirs = glob.glob(data_path + os.sep + "*")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        
        dirs = list()
        for data_dir in data_dirs:
            bdf_filepath = glob.glob(data_dir + os.sep + "*.bdf")
            if not bdf_filepath:
                raise ValueError(f"bdf file not found under {data_path}")
            filename = os.path.split(bdf_filepath[0])[-1]
            trail_info = filename.split('_')
            subject = trail_info[1]
            index = trail_info[3][5:]
            dirs.append({"path": data_dir, "subject": subject, "index": index})
        dirs.sort(key=lambda x: (x['subject'], x['index']))

        return dirs
    
    def video_frame_generator(self, data_dir):
        data_path = data_dir['path']
        subject_name = data_dir['subject']
        trail_index = data_dir['index']
        video_path = glob.glob(data_path + os.sep + "*_C_Section_*.avi")
        if not video_path:
            raise ValueError(f"color video not found under {data_path}")
        video_path = video_path[0]
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
        bdf_file = glob.glob(data_path + os.sep + "*.bdf")
        if not bdf_file:
            raise ValueError(f"bdf file not found under {data_path}")
        bdf_file = pyedflib.EdfReader(bdf_file[0])
        channel_idx = 33
        ecg_signal = bdf_file.readSignal(channel_idx)
        sample_rate = bdf_file.getSampleFrequency(channel_idx)
        annotations = bdf_file.readAnnotations()
        waves = np.array(ecg_signal)
        sq_vec = np.ones_like(waves)
        return waves, sq_vec