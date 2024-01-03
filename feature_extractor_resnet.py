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

# REF https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

logger = logging.getLogger(__name__)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


scaler = transforms.Resize((224, 224))


def build_resnet_model():
    # Load pre-trained ResNet model
    # model = models.resnet34(pretrained=True)
    model = models.resnet18(pretrained=True)
    # Modify the first convolutional layer to handle dynamic input sizes
    # model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # remove the last two layers (avgpool and fc)

    model.avgpool = nn.Identity()
    model.fc = nn.Identity()

    # prevent the model from learning
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    return model


def dump_model_info(model: nn.Module) -> None:
    count, idx = 0, 0
    special_layers = []
    for name, module in model.named_modules():
        if True or isinstance(module, nn.Conv2d):  # and not name.find('downsample') > 0:
            # print(f'{idx} [{count}] = {name} =  {module.weight.shape}')
            print(f'{idx} [{count}] = {name} ')
            count += 1

        if name.find('conv2') > 0:
            special_layers.append(name)
        idx += 1

    print('total conv layers: ', count)
    print('special layers: ', special_layers)

    print("Learnable layers:")
    # Get the list of learnable layers
    learnable_layers = [name for name, param in model.named_parameters() if param.requires_grad]
    # Print the names of learnable layers
    for layer in learnable_layers:
        print(layer)


def get_conv_layers(model: nn.Module) -> list[Conv2d]:
    count, idx = 0, 0
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append(module)
            count += 1
        idx += 1
    print('total conv layers: ', count)
    return conv_layers


def get_layer_by_name(model: nn.Module, target_name: str) -> nn.Conv2d:
    for name, module in model.named_modules():
        if name == target_name:
            return module
    raise Exception(f'layer {target_name} not found')


def get_conv_layer_index_by_name(model: nn.Module, target_name: str):
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if name == target_name:
                return count
            count += 1
    raise Exception(f'layer {target_name} not found')


def get_conv_layer_by_idx(model: nn.Module, idx: int) -> nn.Conv2d:
    count = 0
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            if count == idx:
                return layer
            count += 1
    raise Exception(f'layer {idx} not found')


def visualize_feature(feature, name):
    import matplotlib.pyplot as plt

    if isinstance(feature, torch.Tensor):
        feature = feature.cpu().numpy()

    feature = (feature - feature.min()) / (feature.max() - feature.min())
    feature = (feature * 255).astype(np.uint8)

    layer_viz = feature[0, :, :, :]
    plt.figure(figsize=(30, 30))

    for i, filter in enumerate(layer_viz):
        if i == 16:  # we will visualize only 8x8 blocks from each layer
            break
        plt.subplot(4, 4, i + 1)
        # plt.imshow(filter, cmap='jet')
        plt.imshow(filter, cmap='gray')
        plt.axis("off")

    print(f"Saving layer {name} feature maps...")
    plt.savefig(f"./resnet/layer_{name}.png")
    plt.close()


class FeatureOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, model_input, model_output):
        feature = model_output.detach()
        self.outputs.append(feature)

    def clear(self):
        self.outputs = []


def extract_resnet_features(model: nn.Module, img: Union[torch.Tensor, np.ndarray], layers: list[Conv2d],
                            visualize: bool = False):
    batched_inputs = None
    if isinstance(img, np.ndarray):
        # Convert to N, C, H, W format
        batched_inputs = normalize(to_tensor(img)).unsqueeze(0)
        # batched_inputs = transform(to_tensor(img)).unsqueeze(0)
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
        print(layer)
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

    model = build_resnet_model()
    print('model loaded')
    print(model)

    dump_model_info(model)

    # conv1 ONLY
    targets_resnet50 = ['layer1.0.conv1', 'layer1.1.conv1', 'layer1.2.conv1', 'layer2.0.conv1', 'layer2.1.conv1',
                        'layer2.2.conv1', 'layer2.3.conv1', 'layer3.0.conv1', 'layer3.1.conv1', 'layer3.2.conv1',
                        'layer3.3.conv1', 'layer3.4.conv1', 'layer3.5.conv1', 'layer4.0.conv1', 'layer4.1.conv1',
                        'layer4.2.conv1']

    targets_resnet18_c1_c2 = ['layer1.0.conv1', 'layer1.0.conv2', 'layer1.1.conv1', 'layer1.1.conv2', 'layer2.0.conv1',
                        'layer2.0.conv2', 'layer2.1.conv1', 'layer2.1.conv2', 'layer3.0.conv1', 'layer3.0.conv2',
                        'layer3.1.conv1', 'layer3.1.conv2', 'layer4.0.conv1', 'layer4.0.conv2', 'layer4.1.conv1',
                        'layer4.1.conv2']
    targets_resnet18 =  ['layer1.0.conv2', 'layer1.1.conv2', 'layer2.0.conv2', 'layer2.1.conv2', 'layer3.0.conv2', 'layer3.1.conv2', 'layer4.0.conv2', 'layer4.1.conv2']
    targets = targets_resnet18

    layers = [get_layer_by_name(model, target_name) for target_name in targets]

    layers_idx = [get_conv_layer_index_by_name(model, target_name) for target_name in targets]
    print(layers_idx)

    features = extract_resnet_features(model, img, layers, visualize=True)

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
