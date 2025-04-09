from yacs.config import CfgNode as CN
import yaml
import os

_C = CN()

# Base config files
_C.BASE = ['']

"""
dataloader
"""
_C.DATALOADER = CN()

# video fps in the dataset
_C.DATALOADER.FPS = 30

# name of the dataset
_C.DATALOADER.DATASET = None

# dataset path
_C.DATALOADER.RAW_PATH = None
_C.DATALOADER.CACHED_PATH = None

"""
preprocessing
"""
_C.PRE_PROCESSING = CN()

# face detection and cropping
_C.PRE_PROCESSING.CROP_FACE = CN()
_C.PRE_PROCESSING.CROP_FACE.DO_CROP_FACE = True # do face detection or just use whole frame
_C.PRE_PROCESSING.CROP_FACE.BACKEND = 'MTCNN' # face detection backend
_C.PRE_PROCESSING.CROP_FACE.USE_LARGE_FACE_BOX = True # use larger face bbox than detector output
_C.PRE_PROCESSING.CROP_FACE.LARGE_BOX_COEF = 1.5 # larger face bbox coef
_C.PRE_PROCESSING.CROP_FACE.DO_DYNAMIC_DETECTION = False # use dynamic detection instead of only do detection on first frame
_C.PRE_PROCESSING.CROP_FACE.DYNAMIC_DETECTION_FREQUENCY = 30 # face detection freq in dynamic detection
_C.PRE_PROCESSING.CROP_FACE.USE_MEDIAN_FACE_BOX = False # use median of all the bboxes

# resizing
_C.PRE_PROCESSING.RESIZE = CN()
_C.PRE_PROCESSING.RESIZE.W = 128
_C.PRE_PROCESSING.RESIZE.H = 128

"""
inference
"""
_C.INFERENCE = CN()

# rppg method name
# supported methods: ["unsupervised", "deepphys", "physnet"]
_C.INFERENCE.METHOD_NAME = "unsupervised"

# chunking
_C.INFERENCE.CHUNK = CN()
_C.INFERENCE.CHUNK.DO_CHUNK = True # split one video sequence into chunks
_C.INFERENCE.CHUNK.CHUNK_LENGTH = 300 # if DO_CHUNK is True, this is the chunk length, else chunk_length=video_length
_C.INFERENCE.CHUNK.CHUNK_OVERLAP = True # chunk overlaps each other
_C.INFERENCE.CHUNK.CHUNK_OVERLAP_STEP = 30 # if CHUNK_OVERLAP is True, this is the step between chunks, else step=chunk_length

# batching
_C.INFERENCE.BATCH_SIZE = 1

# input data properties
_C.INFERENCE.DEVICE = "cpu"
_C.INFERENCE.INPUT_FRAME_SIZE = (72, 72) # specify input frame size, None means any frame size is ok
_C.INFERENCE.INPUT_FRAMES_NUM = 128 # specify input number of frames to model, None means any number of frames is ok
_C.INFERENCE.INPUT_TYPE = ['raw', 'standardized', 'diffnormalized']
_C.INFERENCE.LABEL_TYPE = 'raw'
_C.INFERENCE.INPUT_FORMAT = 'NDHWC'

"""
post-processing
"""
_C.POST_PROCESSING = CN()

# bandpass filter on bvp signal
_C.POST_PROCESSING.BANDPASS = True 
_C.POST_PROCESSING.BANDPASS_LOW_FREQ = 45 # in bpm
_C.POST_PROCESSING.BANDPASS_HIGH_FREQ = 150 # in bpm
_C.POST_PROCESSING.FREQ_RESOLUTION = 2 # in bpm

"""
confidence model
"""
_C.CONFIDENCE_MODEL = CN()
_C.CONFIDENCE_MODEL.BANDPASS = True
_C.CONFIDENCE_MODEL.BANDPASS_LOW_FREQ = 45 # in bpm
_C.CONFIDENCE_MODEL.BANDPASS_HIGH_FREQ = 300 # in bpm, consider second harmonic
_C.CONFIDENCE_MODEL.FREQ_RESOLUTION = 2 # in bpm


# update cfg given yaml file
def _update_config_from_file(config, cfg_file):
    config.defrost()

    # load yaml file
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    # build config based on 'BASE' config files
    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )

	# apply current config file
    config.merge_from_file(cfg_file)

    config.freeze()

# get local default cfg given command line args
def get_config_from_file(config_file):
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    _update_config_from_file(config, config_file)

    return config

# get local default cfg
def get_default_config():
    return _C.clone()

# global default cfg
cfg = _C 