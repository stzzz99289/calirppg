#!/bin/bash

# pure
# python preprocess.py --config configs/preprocessing/PURE_preprocessing_72.yaml
# python preprocess.py --config configs/preprocessing/PURE_preprocessing_128.yaml

# ubfc-rPPG
# python preprocess.py --config configs/preprocessing/UBFC-rPPG_preprocessing_72.yaml
# python preprocess.py --config configs/preprocessing/UBFC-rPPG_preprocessing_128.yaml

# iBVP
# python preprocess.py --config configs/preprocessing/iBVP_preprocessing_72.yaml
python preprocess.py --config configs/preprocessing/iBVP_preprocessing_128.yaml

# ubfc-phys
# python preprocess.py --config configs/preprocessing/UBFC-Phys_preprocessing_72.yaml
# python preprocess.py --config configs/preprocessing/UBFC-Phys_preprocessing_128.yaml

# MAHNOB-HCI
# python preprocess.py --config configs/preprocessing/MAHNOB_preprocessing_72.yaml
# python preprocess.py --config configs/preprocessing/MAHNOB_preprocessing_128.yaml