from torch.utils.data import Dataset
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
import time
from math import floor
from tqdm import tqdm
from datetime import datetime

class BaseLoader(Dataset):

    def __init__(self, config_data):
        """Inits dataloader with lists of files.

        Args:
            dataset_name(str): name of the dataloader.
            raw_data_path(string): path to the folder containing all data.
            config_data(CfgNode): data settings(ref:config.py).
        """
        self.config_data = config_data

        # dataloader config
        self.raw_data_path = config_data.DATALOADER.RAW_PATH
        self.cached_path = config_data.DATALOADER.CACHED_PATH
        self.dataset_name = config_data.DATALOADER.DATASET
        self.fps = config_data.DATALOADER.FPS

        # inference config
        self.do_chunk = config_data.INFERENCE.CHUNK.DO_CHUNK
        self.chunk_length = config_data.INFERENCE.CHUNK.CHUNK_LENGTH
        self.chunk_overlap = config_data.INFERENCE.CHUNK.CHUNK_OVERLAP
        self.chunk_overlap_step = config_data.INFERENCE.CHUNK.CHUNK_OVERLAP_STEP

        # preprocessing info
        self.preprocessed_data_info = None
        self.logs = []

        # inference info
        self.chunking_index_table = None
        self.dataset_length = 0

    def __len__(self):
        """Get the length of the dataset.
        """
        return self.dataset_length
    
    def __getitem__(self, index):
        """Returns a clip of video(3,T,W,H) and it's corresponding signals(T)."""
        # get data info
        chunk_idx_global = index
        chunk_idx_local = self.chunking_index_table.iloc[chunk_idx_global]['chunk_idx_local']
        trail_idx = self.chunking_index_table.iloc[chunk_idx_global]['trail_idx']
        chunk_start = self.chunking_index_table.iloc[chunk_idx_global]['chunk_start']
        chunk_end = self.chunking_index_table.iloc[chunk_idx_global]['chunk_end']
        data_info = self.preprocessed_data_info.iloc[trail_idx]

        # load original frames and label
        label_only = False # only load label to save time when frames are not needed
        label = np.load(os.path.join(self.cached_path, data_info['label_path']))
        if not label_only:
            input = np.load(os.path.join(self.cached_path, data_info['input_path']))
        else:
            input = np.zeros((self.chunk_length, self.config_data.PRE_PROCESSING.RESIZE.H, self.config_data.PRE_PROCESSING.RESIZE.W, 9))

        # slice data based on chunk_length and chunk_overlap
        input = np.float32(input[chunk_start:chunk_end])
        label = np.float32(label[chunk_start:chunk_end])
        
        # return item
        # print(f"item idx: {index} from trail {data_info['subject_name']}_{data_info['trail_index']} {chunk_start} ~ {chunk_end}")
        filename = f'{data_info["subject_name"]}_{data_info["trail_index"]}'
        return input, label, filename, chunk_idx_local

    def preprocess(self):
        """Online preprocessing (we do not know the video length in advance)
        """
        # initialize face detection model for preprocessing
        self.face_detection_backend = self.config_data.PRE_PROCESSING.CROP_FACE.BACKEND
        if self.face_detection_backend == "HC":
            # Use OpenCV's Haar Cascade algorithm implementation for face detection
            # This should only utilize the CPU
            self.face_detection_model = cv2.CascadeClassifier('./weights/haarcascade_frontalface_default.xml')
        elif self.face_detection_backend == "MTCNN":
            # Use MTCNN for face detection
            self.face_detection_model = MTCNN(keep_all=True, select_largest=True, device='cuda:0')

        # save config
        os.makedirs(self.cached_path, exist_ok=True)
        self.config_data.dump(stream=open(os.path.join(self.cached_path, "config.yaml"), "w"))

        # preprocessing
        raw_data_dirs = self.get_raw_data_dirs(self.raw_data_path)
        print(f"preprocessing {self.dataset_name} dataset...")
        self.logs = []
        self.preprocessed_data_info = []
        pbar = tqdm(raw_data_dirs, total=len(raw_data_dirs))
        start_time = time.time()
        for data_dir in pbar:
            # get trail info
            trail_path = data_dir['path']
            subject_name = data_dir['subject']
            trail_index = data_dir['index']

            # check if preprocessed data already exists
            self.update_logs(f"===subject {subject_name} trail {trail_index} at {trail_path}===")
            preprocessed_input_path = f"{subject_name}_{trail_index}_input.npy" # path relative to cached_path
            preprocessed_label_path = f"{subject_name}_{trail_index}_label.npy" # path relative to cached_path
            if os.path.exists(os.path.join(self.cached_path, preprocessed_input_path)) \
                and os.path.exists(os.path.join(self.cached_path, preprocessed_label_path)):
                self.update_logs(f"preprocessed data already exists for {subject_name}, trail {trail_index}, skipping...")
                # update preprocessed data info
                self.preprocessed_data_info.append({
                    'input_path': preprocessed_input_path,
                    'label_path': preprocessed_label_path,
                    'subject_name': subject_name,
                    'trail_index': trail_index,
                    'length': len(np.load(os.path.join(self.cached_path, preprocessed_label_path))),
                })
                continue

            # get processed frames and labels
            resized_face_frames = self.get_resized_face_frames(data_dir) # shape: (chunk_length, H, W, 3)
            preprocessed_frames = self.preprocess_frames(resized_face_frames) # shape: (chunk_length, H, W, 9)
            preprocessed_labels, sq_vec = self.get_preprocessed_labels(data_dir, target_length=preprocessed_frames.shape[0]) # labels shape: (chunk_length, 3), sq_vec shape: (chunk_length, )

            # discard frames based on signal quality and do shape check
            del_idx = (sq_vec <= 0.3)
            preprocessed_frames = np.delete(preprocessed_frames, del_idx, axis=0)
            preprocessed_labels = np.delete(preprocessed_labels, del_idx, axis=0)
            sq_vec = np.delete(sq_vec, del_idx, axis=0)
            bvp_len = np.shape(preprocessed_labels)[0]
            frames_len = np.shape(preprocessed_frames)[0]
            assert bvp_len == frames_len, f"bvp len ({bvp_len}) != frames len ({frames_len})"

            # update preprocessed data info
            self.preprocessed_data_info.append({
                'input_path': preprocessed_input_path,
                'label_path': preprocessed_label_path,
                'subject_name': subject_name,
                'trail_index': trail_index,
                'length': len(preprocessed_frames),
            })

            # saving preprocessed data
            preprocessed_frames, preprocessed_labels = np.array(preprocessed_frames), np.array(preprocessed_labels) # shape: (T, W, H, 3), (T, 3)
            self.save_data(preprocessed_frames, 
                            preprocessed_labels, 
                            os.path.join(self.cached_path, preprocessed_input_path), 
                            os.path.join(self.cached_path, preprocessed_label_path),
                            visualize_frames=False, visualize_labels=False)

        # save logs
        end_time = time.time()
        self.update_logs(f"preprocessing completed in {(end_time - start_time):.1f} seconds")
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        logs_path = os.path.join(self.cached_path, f'preprocessing_logs_{timestamp}.txt')
        with open(logs_path, "w") as f:
            f.write('\n'.join(self.logs))
        print(f"preprocessing logs saved to {logs_path}")

        # save preprocessed data paths
        self.preprocessed_data_info = pd.DataFrame(self.preprocessed_data_info)
        self.preprocessed_data_info = self.preprocessed_data_info.sort_values(by=['subject_name', 'trail_index'])
        preprocessed_datainfo_path = os.path.join(self.cached_path, 'preprocessed_datainfo.csv')
        self.preprocessed_data_info.to_csv(preprocessed_datainfo_path, index=False)
        print(f"preprocessed data info saved to {preprocessed_datainfo_path}")

    def save_data(self, frames, bvps, input_path, labels_path, visualize_frames=True, visualize_labels=True):
        """Save data for a single trail.

        Args:
            frames(np.array): video frames.
            bvps(np.array): blood volumne pulse (PPG) labels.
            input_path(str): path to save video frames.
            labels_path(str): path to save labels.
            save_visualization(bool): whether to save frames visualization as avi video.
        """
        # save data
        np.save(input_path, frames)
        np.save(labels_path, bvps)
        self.update_logs(f"preprocessed frames of shape {np.shape(frames)} saved to {input_path}")
        self.update_logs(f"preprocessed label of shape {np.shape(bvps)} saved to {labels_path}") 

        # save visualization for debugging
        if visualize_frames:
            raw_frames = frames[:, :, :, 0:3]
            # standardized_frames = frames[:, :, :, 3:6]
            # diff_normalized_frames = frames[:, :, :, 6:9]
            self.visualize_frames(raw_frames, os.path.join(self.cached_path, input_path.replace('.npy', '_raw.avi'))) 
            # self.visualize_frames(standardized_frames, os.path.join(self.cached_path, input_path.replace('.npy', '_standardized.avi')))
            # self.visualize_frames(diff_normalized_frames, os.path.join(self.cached_path, input_path.replace('.npy', '_diff_normalized.avi')))

        if visualize_labels:
            self.visualize_labels(bvps[:, 0], os.path.join(self.cached_path, labels_path.replace('.npy', '_labels_raw.png')), title='raw')
            self.visualize_labels(bvps[:, 1], os.path.join(self.cached_path, labels_path.replace('.npy', '_labels_standardized.png')), title='standardized')
            self.visualize_labels(bvps[:, 2], os.path.join(self.cached_path, labels_path.replace('.npy', '_labels_diff_normalized.png')), title='diff-normalized')

    def load_preprocessed_data_info(self):
        """Load preprocessed data info.
        """
        # get preprocessed data info dataframe
        preprocessed_datainfo_path = os.path.join(self.cached_path, 'preprocessed_datainfo.csv')
        if not os.path.exists(preprocessed_datainfo_path):
            raise FileNotFoundError(f"preprocessed data info not found at {preprocessed_datainfo_path}")
        self.preprocessed_data_info = pd.read_csv(preprocessed_datainfo_path, 
                                                    dtype={'input_path': str, 'label_path': str, 
                                                    'subject_name': str, 'trail_index': str, 'length': int})

        # build chunking index table and calculated dataset length (total number of chunks)
        self.chunking_index_table = []
        self.dataset_length = 0
        chunk_idx_global = 0
        for trail_index, row in self.preprocessed_data_info.iterrows():
            # calculate number of chunks in each trail
            trail_length = row['length']
            if not self.do_chunk:
                chunk_num = 1 # the whole trail is one chunk
            else:
                if not self.chunk_overlap:
                    chunk_num = floor(trail_length / self.chunk_length) # chunks do not overlap
                else:
                    chunk_num = floor((trail_length - self.chunk_length) / self.chunk_overlap_step) + 1 # chunks overlap
            self.dataset_length += chunk_num
            # build chunking index table (each chunk belongs to which trail, and the start and end frame idx in the trail)
            for chunk_idx_local in range(chunk_num):
                if not self.do_chunk:
                    chunk_start = 0
                    chunk_end = trail_length
                else:
                    if not self.chunk_overlap:
                        chunk_start = chunk_idx_local * self.chunk_length
                        chunk_end = chunk_start + self.chunk_length
                    else:
                        chunk_start = chunk_idx_local * self.chunk_overlap_step
                        chunk_end = chunk_start + self.chunk_length
                self.chunking_index_table.append({
                    'chunk_idx_global': chunk_idx_global,
                        'trail_idx': trail_index,
                        'chunk_idx_local': chunk_idx_local,
                        'chunk_start': chunk_start,
                        'chunk_end': chunk_end,
                })
                chunk_idx_global += 1

        # save chunking index table
        self.chunking_index_table = pd.DataFrame(self.chunking_index_table)
        # chunking_index_table_path = 'chunking_index_table.csv'
        # self.chunking_index_table.to_csv(chunking_index_table_path, index=False)

    def visualize_frames(self, frames, save_path):
        """Visualize frames and save as a video.

        Args:
            frames(np.array): frames of shape (num_frames, height, width, 3).
            save_path(str): path to save the video.
        """
        fourcc = cv2.VideoWriter_fourcc(*'HFYU')
        out = cv2.VideoWriter(save_path, fourcc, self.fps, (frames.shape[2], frames.shape[1]))

        # Global normalization for float frames
        if frames.dtype != np.uint8:
            # Rescale all frames to [0, 1] range using global min and max
            frames_min = frames.min()
            frames_max = frames.max()
            frames = (frames - frames_min) / (frames_max - frames_min + 1e-7)
            # Convert to uint8 range [0, 255]
            frames = (frames * 255).astype(np.uint8)
        
        for frame in frames:
            # Convert from RGB to BGR (OpenCV uses BGR)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()

    def visualize_labels(self, labels, save_path, title=None):
        """Visualize labels and save as a plot figure.

        Args:
            labels(np.array): labels of shape (num_frames, 3).
            save_path(str): path to save the video.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(labels)
        plt.xlabel('local frame idx')
        if title is not None:
            plt.title(title)
        plt.savefig(save_path)
        plt.close()

    def preprocess_frames(self, frames):
        """Preprocess frames.
        Args:
            frames(list[np.array(float)]): frames of shape (H, W, 3).
        Returns:
            preprocessed_frames(list[np.array(float)]): preprocessed frames of shape (H, W, 9). 0~2: raw, 3~5: standardized, 6~8: diff-normalized.
        """
        raw_frames = frames
        standardized_frames = self.standardized_frames(raw_frames)
        diff_normalized_frames = self.diff_normalize_frames(raw_frames)
        preprocessed_frames = np.concatenate([raw_frames, standardized_frames, diff_normalized_frames], axis=-1, dtype=np.float32)
        return preprocessed_frames

    def get_resized_face_frames(self, data_dir):
        """Get preprocessed frames by face detection, cropping and resizing.

        Args:
            data_dir(dict): raw data info of a single trail.
        Returns:
            resized_frames(list[np.array(float)]): Resized and cropped frames
        """
        # load configs
        use_face_detection = self.config_data.PRE_PROCESSING.CROP_FACE.DO_CROP_FACE
        use_dynamic_detection = self.config_data.PRE_PROCESSING.CROP_FACE.DO_DYNAMIC_DETECTION
        detection_freq = self.config_data.PRE_PROCESSING.CROP_FACE.DYNAMIC_DETECTION_FREQUENCY
        use_larger_box = self.config_data.PRE_PROCESSING.CROP_FACE.USE_LARGE_FACE_BOX
        larger_box_coef = self.config_data.PRE_PROCESSING.CROP_FACE.LARGE_BOX_COEF
        use_median_box = self.config_data.PRE_PROCESSING.CROP_FACE.USE_MEDIAN_FACE_BOX
        resized_width = self.config_data.PRE_PROCESSING.RESIZE.W
        resized_height = self.config_data.PRE_PROCESSING.RESIZE.H

        # cropping and detection
        face_region_all = [] # all detected face regions
        resized_frames = [] # all preprocessed frames

        # get first detected face region
        for frame_idx, frame in self.video_frame_generator(data_dir):
            face_region_current, face_detected = self.face_detection(frame, frame_idx, use_larger_box, larger_box_coef)
            if face_detected:
                self.logs.append(f"first face detected at frame {frame_idx}")
                face_region_all.append(face_region_current)
                break

        # cropping and face detection for all frames
        for frame_idx, frame in self.video_frame_generator(data_dir):
            # dynamic face detection, updating face_region_all and face_region_median
            if use_dynamic_detection:
                if frame_idx % detection_freq == 0 and frame_idx > 0: # do face detection every [detection_freq] frames
                    face_region_current, face_detected = self.face_detection(frame, frame_idx, use_larger_box, larger_box_coef)
                    if face_detected:
                        # face detected, append new face region
                        face_region_all.append(face_region_current) 
                    else:
                        # no face detected, append previous face region
                        face_region_all.append(face_region_all[-1])
            
            # Generate a median bounding box based on all detected face regions
            if use_median_box: 
                face_region_median = np.median(np.asarray(face_region_all, dtype='int'), axis=0)

            # frame cropping and resizing
            if use_dynamic_detection:  # use the (i // detection_freq)-th facial region.
                reference_index = frame_idx // detection_freq
            else:  # use the first region obtrained from the first frame.
                reference_index = 0
            if use_face_detection:
                # select current face region
                if use_median_box:
                    face_region = face_region_median
                else:
                    face_region = face_region_all[reference_index]

                # crop frame given face region
                cropped_frame = frame[max(face_region[1], 0):min(face_region[1] + face_region[3], frame.shape[0]),
                        max(face_region[0], 0):min(face_region[0] + face_region[2], frame.shape[1])]
                resized_frame = cv2.resize(cropped_frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
                resized_frames.append(resized_frame)
        
        # convert list to numpy array
        resized_frames = np.array(resized_frames, dtype=np.uint8)

        return resized_frames

    def get_preprocessed_labels(self, data_dir, target_length):
        """Get preprocessed labels by reading raw bvp wave.
        Args:
            data_dir(dict): raw data info of a single trail.
            target_length(int): target length of the preprocessed labels.
        Returns:
            preprocessed_labels(np.array): preprocessed labels of shape (target_length, 3). 0: raw, 1: standardized, 2: diff-normalized.
            sq_vec(np.array): signal quality vector of shape (target_length, ).
        """
        # read raw bvp wave
        labels, sq_vec = self.read_raw_bvp_wave(data_dir)

        # resample ppg signal to match frames
        labels = self.resample_ppg(labels, target_length)
        sq_vec = self.resample_ppg(sq_vec, target_length)

        # preprocess labels
        raw_labels = labels.reshape(-1, 1)
        standardized_labels = self.standardized_label(raw_labels).reshape(-1, 1)
        diff_normalized_labels = self.diff_normalize_label(raw_labels).reshape(-1, 1)
        preprocessed_labels = np.concatenate([raw_labels, standardized_labels, diff_normalized_labels], axis=-1, dtype=np.float32)
        return preprocessed_labels, sq_vec

    def face_detection(self, frame, frame_idx=None, use_larger_box=False, larger_box_coef=1.0):
        """Face detection on a single frame.

        Args:
            frame(np.array): a single frame.
            frame_idx(int): index of the frame.
            backend(str): backend to utilize for face detection.
            use_larger_box(bool): whether to use a larger bounding box on face detection.
            larger_box_coef(float): Coef. of larger box.
        Returns:
            face_box_coor(List[int]): coordinates of face bouding box.
        """
        if self.face_detection_backend == "HC":
            # Computed face_zone(s) are in the form [x_coord, y_coord, width, height]
            # (x,y) corresponds to the top-left corner of the zone
            face_zones = self.face_detection_model.detect(frame)

            if len(face_zones) < 1:
                face_detected = False
                face_box_coor = [0, 0, frame.shape[0], frame.shape[1]]
                self.logs.append(f"Warning: No face is detected at frame {frame_idx}.")
            elif len(face_zones) >= 2:
                # Find the index of the largest face zone
                # The face zones are boxes, so the width and height are the same
                face_detected = True
                max_width_index = np.argmax(face_zones[:, 2])  # Index of maximum width
                face_box_coor = face_zones[max_width_index]
                self.logs.append(f"Warning: More than one faces are detected at frame {frame_idx}. Only cropping the biggest one.")
            else:
                face_detected = True
                face_box_coor = face_zones[0]
        elif self.face_detection_backend == "MTCNN":
            # Computed face_zone(s) are in the form [x0, y0, x1, y1]
            # (x0, y0) corresponds to the top-left corner of the zone
            face_zones, _ = self.face_detection_model.detect(frame)
            
            if face_zones is None or len(face_zones) < 1:
                face_detected = False
                face_box_coor = [0, 0, frame.shape[0], frame.shape[1]]
                self.logs.append(f"Warning: No face is detected at frame {frame_idx}.")
            else:
                # use the largest face zone in the form [x_coord, y_coord, width, height]
                face_detected = True
                face_box_coor = [face_zones[0][0], face_zones[0][1], face_zones[0][2] - face_zones[0][0], face_zones[0][3] - face_zones[0][1]]
                if len(face_zones) > 1:
                    self.logs.append(f"Warning: More than one faces are detected at frame {frame_idx}. Only cropping the biggest one.")
        else:
            raise ValueError("Unsupported face detection backend!")

        if use_larger_box:
            face_box_coor[0] = max(0, face_box_coor[0] - (larger_box_coef - 1.0) / 2 * face_box_coor[2])
            face_box_coor[1] = max(0, face_box_coor[1] - (larger_box_coef - 1.0) / 2 * face_box_coor[3])
            face_box_coor[2] = larger_box_coef * face_box_coor[2]
            face_box_coor[3] = larger_box_coef * face_box_coor[3]

        face_box_coor = [int(x) for x in face_box_coor]
        return face_box_coor, face_detected

    def get_raw_data_dirs(self, raw_data_path):
        """Returns raw data directories under the path.

        Args:
            raw_data_path(str): a list of video_files.
        """
        raise NotImplementedError
    
    def read_raw_bvp_wave(self, data_dir):
        """Read raw bvp wave from data_dir.

        Args:
            data_dir(dict): raw data info of a single trail.
        """
        raise NotImplementedError

    def video_frame_generator(self, data_dir):
        """Returns index, video frame iterated each time.

        Args:
            data_dir(str): data info dictionary.
        """
        raise NotImplementedError
    
    def update_logs(self, log):
        """Update logs.

        Args:
            log(str): log message.
        """
        self.logs.append(log)

    @staticmethod
    def diff_normalize_frames(frames):
        """Calculate discrete difference in video data along the time-axis and nornamize by its standard deviation."""
        n, h, w, c = frames.shape
        diffnormalized_len = n - 1
        diffnormalized_frames = np.zeros((diffnormalized_len, h, w, c), dtype=np.float32)
        diffnormalized_frames_padding = np.zeros((1, h, w, c), dtype=np.float32)
        for j in range(diffnormalized_len):
            diffnormalized_frames[j, :, :, :] = (frames[j + 1, :, :, :] - frames[j, :, :, :]) / (
                    frames[j + 1, :, :, :] + frames[j, :, :, :] + 1e-7)
        diffnormalized_frames = diffnormalized_frames / np.std(diffnormalized_frames)
        diffnormalized_frames = np.append(diffnormalized_frames, diffnormalized_frames_padding, axis=0)
        diffnormalized_frames[np.isnan(diffnormalized_frames)] = 0
        return diffnormalized_frames

    @staticmethod
    def diff_normalize_label(label):
        """Calculate discrete difference in labels along the time-axis and normalize by its standard deviation."""
        diff_label = np.diff(label, axis=0)
        diffnormalized_label = diff_label / np.std(diff_label)
        diffnormalized_label = np.append(diffnormalized_label, np.zeros(1), axis=0)
        diffnormalized_label[np.isnan(diffnormalized_label)] = 0
        return diffnormalized_label

    @staticmethod
    def standardized_frames(frames):
        """Z-score standardization for video data."""
        frames = frames - np.mean(frames)
        frames = frames / np.std(frames)
        frames[np.isnan(frames)] = 0
        return frames

    @staticmethod
    def standardized_label(label):
        """Z-score standardization for label signal."""
        label = label - np.mean(label)
        label = label / np.std(label)
        label[np.isnan(label)] = 0
        return label

    @staticmethod
    def resample_ppg(input_signal, target_length):
        """Samples a PPG sequence into specific length."""
        return np.interp(
            np.linspace(1, input_signal.shape[0], target_length), 
            np.linspace(1, input_signal.shape[0], input_signal.shape[0]), 
            input_signal
            )