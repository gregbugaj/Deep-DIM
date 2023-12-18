# MPViT : Multi-Path Vision Transformer for Dense Prediction
# https://arxiv.org/pdf/2112.11010.pdf
# https://medium.com/@hirotoschwert/digging-into-detectron-2-part-2-dd6e8b0526e

import argparse
import logging
import os
from typing import Dict, List, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.structures import ImageList

from ditod import add_vit_config

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
    model.eval()
    logger.info("Model:\n{}".format(model))

    return model


def preprocess_image(
    batched_inputs: List[torch.Tensor],
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
):
    """
    Normalize, pad and batch the input images.
    """
    # backbone.size_divisibility: 32
    # backbone.padding_constraints: {'square_size': 0}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    images = [x.to(device) for x in batched_inputs]
    # images = [(x - pixel_mean) / pixel_std for x in images]
    images = ImageList.from_tensors(
        images,
        32,
    )

    return images


def extract_vit_features(backbone_cfg: Dict, img: Union[torch.Tensor, np.ndarray]):
    backbone = backbone_cfg["backbone"]
    pixel_mean = backbone_cfg["pixel_mean"]
    pixel_std = backbone_cfg["pixel_std"]

    print("Extracting features from image")
    print(backbone.output_shape())
    print("backbone.size_divisibility : ", backbone.size_divisibility)
    print("backbone.padding_constraints : ", backbone.padding_constraints)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if isinstance(img, np.ndarray):
        # Convert to N, C, H, W format
        batched_inputs = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    elif isinstance(img, torch.Tensor):
        # check if the image is in N, C, H, W format
        if len(img.shape) == 3:  # C, H, W
            batched_inputs = img.unsqueeze(0)
        elif len(img.shape) == 4:  # N, C, H, W
            batched_inputs = img
        else:
            raise Exception("Unsupported image shape")
    else:
        raise Exception("Unsupported image type")

    print(batched_inputs.shape)
    images = preprocess_image(
        batched_inputs,
        torch.tensor(pixel_mean).view(-1, 1, 1),
        torch.tensor(pixel_std).view(-1, 1, 1),
    )
    print(images.tensor.shape)

    # 'p2', 'p3', 'p4', 'p5', 'p6'
    fig = plt.figure(figsize=(32, 18))
    # setting values to rows and column variables
    rows = 2
    columns = 3

    # original image
    fig.add_subplot(rows, columns, 1)
    plt.imshow(images.tensor[0].permute(1, 2, 0).cpu().numpy())
    plt.axis('on')
    plt.title("input image")

    with torch.no_grad():
        features = backbone(images.tensor)
        print(features.keys())
        idx = 0
        for k, v in features.items():
            # feature format is N, C, H, W where C is the number of channels (feature maps) = 256
            print(f"output[{k}].shape->", v.shape)
            v_ = v[:, 255].cpu().numpy()
            v_ = (v_ - v_.min()) / (v_.max() - v_.min())
            v_ = (v_ * 255).astype(np.uint8)
            v_ = np.asarray(v_.clip(0, 255), dtype=np.uint8).transpose((1, 2, 0))
            cv2.imwrite(k + '.png', v_)

            fig.add_subplot(rows, columns, idx + 2)
            plt.imshow(v_, cmap='jet')
            plt.axis('on')
            plt.title(k)
            idx += 1

    print(f"Saving  feature maps...")
    plt.savefig(f"features.png")
    plt.close()

    return features


def main(args: argparse.Namespace):
    print("Extracting features from image")

    if args.image_path is None:
        raise Exception("No image path provided")

    image_path = os.path.expanduser(args.image_path)
    if not os.path.exists(image_path):
        raise Exception("Image path does not exist : ", image_path)

    img = cv2.imread(os.path.expanduser(image_path))
    backbone_cfg = build_backbone_config(args)

    return extract_vit_features(backbone_cfg, img)


def build_backbone_config(args: argparse.Namespace) -> Dict[str, object]:
    cfg = setup_cfg(args)
    model = build_model_from_config(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    print("Model loaded from ", cfg.MODEL.WEIGHTS)
    print(model)
    backbone_cfg = {
        "backbone": model.backbone,
        "input_format": cfg.INPUT.FORMAT,
        "pixel_mean": cfg.MODEL.PIXEL_MEAN,
        "pixel_std": cfg.MODEL.PIXEL_STD,
    }
    return backbone_cfg


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
    # set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = get_parser()
    args = parser.parse_args()
    main(args)
