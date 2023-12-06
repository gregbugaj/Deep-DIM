import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os, glob


# REF https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

model = models.resnet50(pretrained=True)
layer = model._modules.get('avgpool')
# layer = model._modules.get('layer1.0.conv1')
model.eval()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()
scaler = transforms.Resize((224, 224))

print('model loaded')
print(model)

for child in model.named_children():
    print(child)
    print('-------------------')

# list all conv layers

# list all conv layers
count = 0
idx = 0
target_name = 'layer1.2.conv3'

for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d) and not name.find('downsample') > 0:
        print(f'{idx} = {name}')
        count += 1
    if name == target_name:
        layer = module
    idx += 1

print('total conv layers: ', count)
 
# get layer by name
# layer = model._modules.get('layer')

def visualize_feature(feature, name):
    import matplotlib.pyplot as plt
    # feature = feature.cpu().numpy()

    # feature = (feature - feature.min()) / (feature.max() - feature.min())
    # feature = (feature * 255).astype(np.uint8)

    layer_viz = feature[0, :, :, :]
    plt.figure(figsize=(30, 30))

    for i, filter in enumerate(layer_viz):
        if i == 16: # we will visualize only 8x8 blocks from each layer
            break
        plt.subplot(4, 4, i + 1)
        plt.imshow(filter, cmap='jet')
        plt.axis("off")

    print(f"Saving layer {name} feature maps...")
    plt.savefig(f"./resnet/layer_{name}_{i}.png")
    plt.close()


def get_vector(image_name):
    img = Image.open(image_name).convert('RGB')
    print(img.size)
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    # t_img = Variable(normalize(to_tensor(img)).unsqueeze(0))
    
    my_embedding = torch.zeros(1,2048,1,1)
    my_embedding = torch.zeros(1, 64, 112, 112)
    my_embedding = torch.zeros(1, 256, 56, 56)

    def copy_data(m, i, o):
        print("out ", o.shape)        
        my_embedding.copy_(o.data)
    h = layer.register_forward_hook(copy_data)
    model(t_img)
    h.remove()
    return my_embedding.numpy()

vec1 = get_vector(os.path.expanduser('~/dev/Deep-DIM/RMSdata/00-1.png'))
print(vec1.shape)

visualize_feature(vec1, target_name)
