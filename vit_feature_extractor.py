# MPViT : Multi-Path Vision Transformer for Dense Prediction
# https://arxiv.org/pdf/2112.11010.pdf


import argparse
import glob
import logging
import os
import sys
import torch
import cv2

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn import Module
from ditod import add_vit_config

from detectron2.utils.visualizer import ColorMode, Visualizer

from detectron2.config import get_cfg
from detectron2.modeling import build_model

from ditod import add_vit_config

import argparse

logger = logging.getLogger(__name__)

def setup_cfg(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # add_coat_config(cfg)
    add_vit_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    cfg.MODEL.DEVICE = device

    cfg.freeze()
    # default_setup(cfg, args)
    return cfg


def build_model_from_config(cfg):
    """
    Returns:
        torch.nn.Module:

    It now calls :func:`detectron2.modeling.build_model`.
    Overwrite it if you'd like a different model.
    """
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))

    return model


def extract_vit_features(model, image_path):
    # Your code to extract features from the image goes here
    pass


def main(args):
    print("Extracting features from image")

    # Step 1: instantiate config
    cfg = setup_cfg(args)
    print(cfg)
    model = build_model_from_config(cfg)    
    if args.image_path is None:
        print("No image path provided")
        return

    args.image_path = os.path.expanduser(args.image_path)    
    extract_vit_features(model, args.image_path)


def get_parser():
    parser = argparse.ArgumentParser(description="DIT Feature Extractor")
    parser.add_argument(
        "--image_path",
        help="Path to input image",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_path",
        help="Name of the output visualization directory.",
        type=str,
    )
    parser.add_argument(
        "--config-file",
        default="configs/mask_rcnn_dit_base.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    return parser


# EOB
#  python ./vit_feature_extractor.py --config-file configs/mask_rcnn_dit_base.yaml  --image_path /home/gbugaj/tmp/demo --output_path /tmp/dit --opts  MODEL.WEIGHTS ~/models/dit_eob_detection/tuned-01/model_0005999.pth


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)    

