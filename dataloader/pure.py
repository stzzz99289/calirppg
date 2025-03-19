"""The dataloader for PURE datasets.

Details for the PURE Dataset see https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure
If you use this dataset, please cite the following publication:
Stricker, R., Müller, S., Gross, H.-M.
Non-contact Video-based Pulse Rate Measurement on a Mobile Service Robot
in: Proc. 23st IEEE Int. Symposium on Robot and Human Interactive Communication (Ro-Man 2014), Edinburgh, Scotland, UK, pp. 1056 - 1062, IEEE 2014
"""
import glob
import os
import cv2
import json
import numpy as np
from dataloader.base import BaseLoader


class PURELoader(BaseLoader):
    """The data loader for the PURE dataset."""

    def __init__(self, config_data):
        """Initializes an PURE dataloader.
            data_path should be "RawData" for below dataset structure:
            -----------------
                RawData/
                |   |-- 01-01/
                |      |-- 01-01/
                |      |-- 01-01.json
                |   |-- 01-02/
                |      |-- 01-02/
                |      |-- 01-02.json
                |...
                |   |-- ii-jj/
                |      |-- ii-jj/
                |      |-- ii-jj.json
            -----------------
        """
        super().__init__(config_data)

    def get_raw_data_dirs(self, data_path):
        data_dirs = glob.glob(data_path + os.sep + "*-*")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        
        dirs = list()
        for data_dir in data_dirs:
            subject_trail_val = os.path.split(data_dir)[-1].replace('-', '')
            index = subject_trail_val[2:4]
            subject = subject_trail_val[0:2]
            dirs.append({"path": data_dir, "subject": subject, "index": index})
        dirs.sort(key=lambda x: (x['subject'], x['index']))

        return dirs
    
    def video_frame_generator(self, data_dir):
        data_path = data_dir['path']
        subject_name = data_dir['subject']
        trail_index = data_dir['index']
        search_png_path = os.path.join(data_path, f'{subject_name}-{trail_index}', '*.png')
        all_png = sorted(glob.glob(search_png_path))
        for frame_idx, bmp_path in enumerate(all_png):
            img = cv2.imread(bmp_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            yield frame_idx, img

    def read_raw_bvp_wave(self, data_dir):
        data_path = data_dir['path']
        subject_name = data_dir['subject']
        trail_index = data_dir['index']
        bvp_file = os.path.join(data_path, f'{subject_name}-{trail_index}.json')
        with open(bvp_file, "r") as f:
            labels = json.load(f)
            waves = [label["Value"]["waveform"]
                     for label in labels["/FullPackage"]]
            waves = np.asarray(waves)
            sq_vec = np.ones_like(waves)
        return waves, sq_vec