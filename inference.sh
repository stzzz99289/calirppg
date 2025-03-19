#!/bin/bash

python inference.py --config configs/PURE_inference.yaml

python inference.py --config configs/UBFC-rPPG_inference.yaml

# too much time, need another machine
# python inference.py --config configs/UBFC-Phys_inference.yaml

# python inference.py --config configs/iBVP_inference.yaml