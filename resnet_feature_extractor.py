import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.models
from PIL import Image
from torch.autograd import Variable
from torchvision import models, transforms

# REF https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

# set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def build_resnet_model():
    # Load pre-trained ResNet-50
    model = models.resnet50(pretrained=True)
    # Modify the first convolutional layer to handle dynamic input sizes
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.avgpool = nn.Identity()
    model.fc = nn.Identity()

    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    return model


model = build_resnet_model()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

print('model loaded')
print(model)

target_name = 'layer1.0.conv3'
# target_name = 'conv1'

def dump_model_info():
    count = 0
    idx = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):  # and not name.find('downsample') > 0:
            print(f'{idx} [{count}] = {name} = {module.weight.shape}')
            count += 1
        idx += 1
    print('total conv layers: ', count)


def get_layer_by_name(target_name: str) -> nn.Conv2d:
    for name, module in model.named_modules():
        if name == target_name:
            return module
    raise Exception(f'layer {target_name} not found')


def get_layer_by_idx(idx: int) -> nn.Conv2d:
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if count == idx:
                return module
            count += 1
    raise Exception(f'layer {idx} not found')


def visualize_feature(feature, name):
    import matplotlib.pyplot as plt

    # feature = feature.cpu().numpy()
    # feature = (feature - feature.min()) / (feature.max() - feature.min())
    # feature = (feature * 255).astype(np.uint8)

    layer_viz = feature[0, :, :, :]
    plt.figure(figsize=(30, 30))

    for i, filter in enumerate(layer_viz):
        if i == 16:  # we will visualize only 8x8 blocks from each layer
            break
        plt.subplot(4, 4, i + 1)
        plt.imshow(filter, cmap='jet')
        plt.axis("off")

    print(f"Saving layer {name} feature maps...")
    plt.savefig(f"./resnet/layer_{name}_{i}.png")
    plt.close()


layer = get_layer_by_name(target_name)


def get_vector(image_name):
    img = Image.open(image_name).convert('RGB')
    print(img.size)
    # t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    t_img = Variable(normalize(to_tensor(img)).unsqueeze(0))

    # my_embedding = torch.zeros(1, 64, 128, 132)

    print("expected ", layer)
    input_size = t_img.shape
    print("Input size:", input_size)

    def copy_data(module, input, output):
        print("feature : ", output.shape)
        # print("my_embedding ", my_embedding.shape)
        feature = output.detach().numpy()
        visualize_feature(feature, f"h-{target_name}")
        # my_embedding.copy_(output.data)

    h = layer.register_forward_hook(copy_data)
    model(t_img)
    h.remove()

    # return my_embedding.numpy()


vec1 = get_vector(os.path.expanduser('~/dev/Deep-DIM/RMSdata/00-2.png'))

# visualize_feature(vec1, target_name)
