import torch

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


class FeatureOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, model_input, model_output):
        feature = model_output.detach()
        self.outputs.append(feature)

    def clear(self):
        self.outputs = []

def dump_model_info(model: nn.Module) -> None:
    count, idx = 0, 0
    special_layers = []
    for name, module in model.named_modules():
        if True or isinstance(module, nn.Conv2d):  # and not name.find('downsample') > 0:
            # print(f'{idx} [{count}] = {name} =  {module.weight.shape}')
            print(f'{idx} [{count}] = {name} ')
            count += 1

        if name.find('conv1') > 0:
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
    plt.savefig(f"./densenet/layer_{name}.png")
    plt.close()
