import argparse
import logging
import os
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Conv2d
from torchvision import models, transforms
from PIL import Image

from feature_util import FeatureOutput, visualize_feature, dump_model_info, get_layer_by_name, \
    get_conv_layer_index_by_name

# REF https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

logger = logging.getLogger(__name__)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()


# scaler = transforms.Resize((224, 224))


def build_densenet_model():
    # Load pre-trained ResNet-50
    # resnet34
    # model = models.resnet50(pretrained=True)
    model = models.densenet121(pretrained=True)
    # Modify the first convolutional layer to handle dynamic input sizes
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

    model.avgpool = nn.Identity()
    model.fc = nn.Identity()

    # for param in model.parameters():
    #     param.requires_grad = False

    model.eval()
    return model


def extract_densenet_features(model: nn.Module, img: Union[torch.Tensor, np.ndarray], layers: list[Conv2d],
                              visualize: bool = False):
    batched_inputs = None
    if isinstance(img, np.ndarray):
        # Convert to N, C, H, W format
        batched_inputs = normalize(to_tensor(img)).unsqueeze(0)
        # batched_inputs = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    elif isinstance(img, torch.Tensor):
        # If the image is already a tensor we don't normalize it as it should normalized
        # check if the image is in N, C, H, W format
        if len(img.shape) == 3:  # C, H, W
            batched_inputs = img.unsqueeze(0)
        elif len(img.shape) == 4:  # N, C, H, W
            batched_inputs = img
        else:
            raise Exception("Unsupported image shape")
    else:
        raise Exception("Unsupported image type")

    # batched_inputs = Variable(normalize(to_tensor(img)).unsqueeze(0))
    # batched_inputs = normalize(to_tensor(img)).unsqueeze(0)
    print("Input size:", batched_inputs.shape)

    save_output = FeatureOutput()
    hook_handles = []

    for layer in layers:
        handle = layer.register_forward_hook(save_output)
        hook_handles.append(handle)

    _ = model(batched_inputs)

    for h in hook_handles:
        h.remove()

    print("Feature count : ", len(save_output.outputs))
    if visualize:
        for i, feature in enumerate(save_output.outputs):
            print(f"Feature {i} shape : ", feature.shape)
            visualize_feature(feature, f"{i}")

    return save_output.outputs


def main(args: argparse.Namespace):
    print("Extracting features from image")

    if args.image_path is None:
        raise Exception("No image path provided")

    image_path = os.path.expanduser(args.image_path)
    if not os.path.exists(image_path):
        raise Exception("Image path does not exist : ", image_path)

    img = cv2.imread(os.path.expanduser(image_path))
    #
    # img = Image.open(image_path).convert('RGB')
    # img = np.array(img)

    model = build_densenet_model()
    print('model loaded')
    print(model)

    dump_model_info(model)

    targets = ['features.denseblock1.denselayer1.conv1', 'features.denseblock1.denselayer2.conv1', 'features.denseblock1.denselayer3.conv1', 'features.denseblock1.denselayer4.conv1', 'features.denseblock1.denselayer5.conv1', 'features.denseblock1.denselayer6.conv1', 'features.denseblock2.denselayer1.conv1', 'features.denseblock2.denselayer2.conv1', 'features.denseblock2.denselayer3.conv1', 'features.denseblock2.denselayer4.conv1', 'features.denseblock2.denselayer5.conv1', 'features.denseblock2.denselayer6.conv1', 'features.denseblock2.denselayer7.conv1', 'features.denseblock2.denselayer8.conv1', 'features.denseblock2.denselayer9.conv1', 'features.denseblock2.denselayer10.conv1', 'features.denseblock2.denselayer11.conv1', 'features.denseblock2.denselayer12.conv1', 'features.denseblock3.denselayer1.conv1', 'features.denseblock3.denselayer2.conv1', 'features.denseblock3.denselayer3.conv1', 'features.denseblock3.denselayer4.conv1', 'features.denseblock3.denselayer5.conv1', 'features.denseblock3.denselayer6.conv1', 'features.denseblock3.denselayer7.conv1', 'features.denseblock3.denselayer8.conv1', 'features.denseblock3.denselayer9.conv1', 'features.denseblock3.denselayer10.conv1', 'features.denseblock3.denselayer11.conv1', 'features.denseblock3.denselayer12.conv1', 'features.denseblock3.denselayer13.conv1', 'features.denseblock3.denselayer14.conv1', 'features.denseblock3.denselayer15.conv1', 'features.denseblock3.denselayer16.conv1', 'features.denseblock3.denselayer17.conv1', 'features.denseblock3.denselayer18.conv1', 'features.denseblock3.denselayer19.conv1', 'features.denseblock3.denselayer20.conv1', 'features.denseblock3.denselayer21.conv1', 'features.denseblock3.denselayer22.conv1', 'features.denseblock3.denselayer23.conv1', 'features.denseblock3.denselayer24.conv1', 'features.denseblock4.denselayer1.conv1', 'features.denseblock4.denselayer2.conv1', 'features.denseblock4.denselayer3.conv1', 'features.denseblock4.denselayer4.conv1', 'features.denseblock4.denselayer5.conv1', 'features.denseblock4.denselayer6.conv1', 'features.denseblock4.denselayer7.conv1', 'features.denseblock4.denselayer8.conv1', 'features.denseblock4.denselayer9.conv1', 'features.denseblock4.denselayer10.conv1', 'features.denseblock4.denselayer11.conv1', 'features.denseblock4.denselayer12.conv1', 'features.denseblock4.denselayer13.conv1', 'features.denseblock4.denselayer14.conv1', 'features.denseblock4.denselayer15.conv1', 'features.denseblock4.denselayer16.conv1']
    layers = [get_layer_by_name(model, target_name) for target_name in targets]

    # layers = get_conv_layers(model)
    layers_idx = [get_conv_layer_index_by_name(model, target_name) for target_name in targets]
    print(layers_idx)

    features = extract_densenet_features(model, img, layers, visualize=True)

    return features


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

    return parser


if __name__ == "__main__":
    # set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = get_parser()
    args = parser.parse_args()
    main(args)
