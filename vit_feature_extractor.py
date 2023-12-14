# MPViT : Multi-Path Vision Transformer for Dense Prediction
# https://arxiv.org/pdf/2112.11010.pdf


import argparse
import glob
import logging
import os
import sys
from typing import List, Dict

import torch
import cv2

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from detectron2.structures import ImageList
from torch.nn import Module
from ditod import add_vit_config
from detectron2.modeling.backbone import build_backbone
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


def preprocess_image(batched_inputs: List[torch.Tensor], pixel_mean: torch.Tensor, pixel_std: torch.Tensor):
    """
    Normalize, pad and batch the input images.
    """
    # backbone.size_divisibility: 32
    # backbone.padding_constraints: {'square_size': 0}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    images = [x.to(device) for x in batched_inputs]
    images = [(x - pixel_mean) / pixel_std for x in images]
    images = ImageList.from_tensors(
        images,
        32,
        padding_constraints={'square_size': 0},
    )
    return images


def extract_vit_features(backbone_cfg: Dict, image_path: str):
    backbone = backbone_cfg["backbone"]
    pixel_mean = backbone_cfg["pixel_mean"]
    pixel_std = backbone_cfg["pixel_std"]

    print("Extracting features from image")
    print(backbone.output_shape())
    print("backbone.size_divisibility : ", backbone.size_divisibility)
    print("backbone.padding_constraints : ", backbone.padding_constraints)

    backbone.eval()
    img = Image.open(os.path.expanduser(image_path)).convert('RGB')


    # Convert to N, C, H, W format
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    batched_inputs = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    print(batched_inputs.shape)

    images = preprocess_image(batched_inputs, torch.tensor(pixel_mean).view(-1, 1, 1),
                              torch.tensor(pixel_std).view(-1, 1, 1))
    print(images.tensor.shape)

    # 'p2', 'p3', 'p4', 'p5', 'p6'
    features = backbone(images.tensor)
    print(features.keys())
    print(features)


def main(args):
    print("Extracting features from image")

    # Step 1: instantiate config
    cfg = setup_cfg(args)
    print(cfg)
    backbone = build_backbone(cfg)

    model = build_model_from_config(cfg)
    if args.image_path is None:
        print("No image path provided")
        return

    args.image_path = os.path.expanduser(args.image_path)
    # extract_vit_features(model, args.image_path)

    backbone_cfg = {
        "backbone": backbone,
        "input_format": cfg.INPUT.FORMAT,
        "pixel_mean": cfg.MODEL.PIXEL_MEAN,
        "pixel_std": cfg.MODEL.PIXEL_STD,
    }

    print(backbone_cfg)
    extract_vit_features(backbone_cfg, args.image_path)


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
