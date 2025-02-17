from __future__ import division, print_function

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
from torchvision import transforms

from DIM import *
from utils import all_sample_iou, plot_success_curve
from feature_extractor_vit import build_backbone_config, extract_vit_features

matplotlib.use('Agg')


class Featex:
    def __init__(self, backbone_cfg, use_cuda, layer1, layer2, layer3):
        self.use_cuda = use_cuda
        self.feature1 = None
        self.feature2 = None
        self.feature3 = None
        self.U1 = None
        self.U2 = None
        self.U3 = None

        self.backbone_cfg = backbone_cfg
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3

    def save_feature1(self, module, input, output):
        self.feature1 = output.detach()
        self.visualize_feature(self.feature1, 'feature1')

    def save_feature2(self, module, input, output):
        self.feature2 = output.detach()
        self.visualize_feature(self.feature2, 'feature2')

    def save_feature3(self, module, input, output):
        self.feature3 = output.detach()
        self.visualize_feature(self.feature3, 'feature3')

    def visualize_feature(self, feature, name):
        import matplotlib.pyplot as plt

    def __call__(self, input, mode='normal'):
        channel = 64
        if self.use_cuda:
            input = input.cuda()

        features = extract_vit_features(self.backbone_cfg, input)
        self.feature1 = features.get(f'p{self.layer1}')
        self.feature2 = features.get(f'p{self.layer2}')
        self.feature3 = features.get(f'p{self.layer3}')

        if channel < self.feature1.shape[1]:
            reducefeature1, self.U1 = runpca(self.feature1, channel, self.U1)
        else:
            reducefeature1 = self.feature1
        if channel < self.feature2.shape[1]:
            reducefeature2, self.U2 = runpca(self.feature2, channel, self.U2)
        else:
            reducefeature2 = self.feature2
        if channel < self.feature3.shape[1]:
            reducefeature3, self.U3 = runpca(self.feature3, channel, self.U3)
        else:
            reducefeature3 = self.feature3

        if False:  # mode == 'big':
            # resize feature1 to the same size of feature2
            reducefeature1 = F.interpolate(
                reducefeature1,
                size=(self.feature3.size()[2], self.feature3.size()[3]),
                mode='bilinear',
                align_corners=True,
            )
            reducefeature2 = F.interpolate(
                reducefeature2,
                size=(self.feature3.size()[2], self.feature3.size()[3]),
                mode='bilinear',
                align_corners=True,
            )

        reducefeature2 = F.interpolate(
            reducefeature2,
            size=(self.feature1.size()[2], self.feature1.size()[3]),
            mode='bilinear',
            align_corners=True,
        )
        reducefeature3 = F.interpolate(
            reducefeature3,
            size=(self.feature1.size()[2], self.feature1.size()[3]),
            mode='bilinear',
            align_corners=True,
        )

        return torch.cat((reducefeature1, reducefeature2, reducefeature3), dim=1)


def runpca(x, components, U):
    whb = x.squeeze(0).permute(1, 2, 0).cpu().numpy()
    shape = whb.shape
    raw = whb.reshape((shape[0] * shape[1], shape[2]))
    X_norm, mu, sigma = featureNormalize(raw)
    if U is None:
        Sigma = np.dot(np.transpose(X_norm), X_norm) / raw.shape[0]
        U, S, V = np.linalg.svd(Sigma)
    Z = projectData(X_norm, U, components)
    return (
        torch.tensor(Z.reshape((shape[0], shape[1], components)))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .cuda(),
        U,
    )


def featureNormalize(X):
    n = X.shape[1]

    sigma = np.zeros((1, n))
    mu = np.zeros((1, n))
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    for i in range(n):
        X[:, i] = (X[:, i] - mu[i]) / sigma[i]
    return X, mu, sigma


def projectData(X_norm, U, K):
    Z = np.zeros((X_norm.shape[0], K))

    U_reduce = U[:, 0:K]
    Z = np.dot(X_norm, U_reduce)
    return Z


def read_gt(file_path):
    with open(file_path) as IN:
        x, y, w, h = [eval(i) for i in IN.readline().strip().split(',')]
    return x, y, w, h


def apply_DIM(I_row, SI_row, template_bbox, pad, pad1, image, numaddtemplates):
    I = preprocess(I_row)
    SI = preprocess(SI_row)
    template, oddTbox = imcrop_odd(I, template_bbox, True)
    targetKeypoints = [
        oddTbox[1] + (oddTbox[3] - 1) / 2,
        oddTbox[0] + (oddTbox[2] - 1) / 2,
    ]
    addtemplates = extract_additionaltemplates(
        I, template, numaddtemplates, np.array([targetKeypoints])
    )
    if len(addtemplates):
        templates = torch.cat((template, addtemplates), 0)
    else:
        templates = template
    print('Numtemplates=', len(templates))
    print('Preprocess done,start matching...')
    similarity = DIM_matching(SI, templates, 10)[
        pad[0] : pad[0] + I.shape[2], pad[1] : pad[1] + I.shape[3]
    ]
    # post processing
    similarity = cv2.resize(similarity, (image.shape[1], image.shape[0]))
    scale = 0.025
    region = (
        torch.from_numpy(
            ellipse(round(max(1, scale * pad1[1])), round(max(1, scale * pad1[0])))
        )
        .type(torch.FloatTensor)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    similarity = (
        conv2_same(torch.from_numpy(similarity).unsqueeze(0).unsqueeze(0), region)
        .squeeze()
        .numpy()
    )
    return similarity


def model_eval(backbone_cfg, layer1, layer2, layer3, file_dir, use_cuda):
    if not os.path.exists(
        'results/'
        + file_dir
        + '/'
        + str(layer1)
        + '_'
        + str(layer2)
        + '_'
        + str(layer3)
    ):
        os.makedirs(
            'results/'
            + file_dir
            + '/'
            + str(layer1)
            + '_'
            + str(layer2)
            + '_'
            + str(layer3)
        )
    featex = Featex(backbone_cfg, use_cuda, layer1, layer2, layer3)

    gt_list, pd_list = [], []
    num_samples = len(img_path) // 2

    print("num_samples: ", num_samples)

    # bar = progressbar.ProgressBar(max_value=num_samples)
    for idx in range(num_samples):
        # load image and ground truth
        template_raw = cv2.imread(img_path[2 * idx])[..., ::-1]
        # template_raw = apply_lbp(template_raw)

        template_bbox = read_gt(gt[2 * idx])
        x, y, w, h = [int(round(t)) for t in template_bbox]
        template_plot = cv2.rectangle(
            template_raw.copy(), (x, y), (x + w, y + h), (0, 255, 0), 2
        )
        image = cv2.imread(img_path[2 * idx + 1])[..., ::-1]
        # image = apply_lbp(image)

        # cv2.imwrite('imageon.png',X[0][0].cpu().numpy()*255)

        image_gt = read_gt(gt[2 * idx + 1])
        root = 'results/' + file_dir + '/{m}/{n}.txt'
        if os.path.exists(
            root.format(
                n=idx + 1, m=str(layer1) + ':' + str(layer2) + ':' + str(layer3)
            )
        ):
            f = open(
                root.format(
                    n=idx + 1, m=str(layer1) + ':' + str(layer2) + ':' + str(layer3)
                ),
                'r',
            )
            image_pd = tuple([float(i) for i in f.read().split(',')])
            f.close()
            gt_list.append(image_gt)
            pd_list.append(image_pd)
            continue
        # bar.update(idx + 1)
        x_gt, y_gt, w_gt, h_gt = [int(round(t)) for t in image_gt]
        image_plot = cv2.rectangle(
            image.copy(), (x_gt, y_gt), (x_gt + w_gt, y_gt + h_gt), (0, 255, 0), 2
        )

        image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize(
                #     mean=[0.127, 0.127, 0.127],
                #     std=[0.127, 0.127, 0.127],
                # ),
            ]
        )

        T_image = image_transform(template_raw.copy()).unsqueeze(0)
        T_search_image = image_transform(image.copy()).unsqueeze(0)

        # Section 4.2 in the paper

        if 0 <= layer1 <= 4:
            a = 1
        if 4 < layer1 <= 9:
            a = 2
        if 9 < layer1 <= 18:
            a = 4
        if 18 < layer1 <= 27:
            a = 8
        if 27 < layer1 <= 36:
            a = 16
        if 0 <= layer3 <= 4:
            b = 1
        if 4 < layer3 <= 9:
            b = 2
        if 9 < layer3 <= 18:
            b = 4
        if 18 < layer3 <= 27:
            b = 8
        if 27 < layer3 <= 36:
            b = 16

        if w * h <= 4000:
            I_feat = featex(T_image)
            SI_feat = featex(T_search_image)
            resize_bbox = [i / a for i in template_bbox]
        else:
            I_feat = featex(T_image, 'big')
            SI_feat = featex(T_search_image, 'big')
            resize_bbox = [i / b for i in template_bbox]

        print(' ')
        print('Feature extraction done.')
        pad1 = [int(round(t)) for t in (template_bbox[3], template_bbox[2])]
        pad2 = [int(round(t)) for t in (resize_bbox[3], resize_bbox[2])]

        SI_pad = torch.from_numpy(
            np.pad(
                SI_feat.cpu().numpy(),
                ((0, 0), (0, 0), (pad2[0], pad2[0]), (pad2[1], pad2[1])),
                'symmetric',
            )
        )
        similarity = apply_DIM(I_feat, SI_pad, resize_bbox, pad2, pad1, image, 5)

        ptx, pty = np.where(similarity == np.amax(similarity))
        image_pd = tuple(
            [
                pty[0] + 1 - (odd(template_bbox[2]) - 1) / 2,
                ptx[0] + 1 - (odd(template_bbox[3]) - 1) / 2,
                template_bbox[2],
                template_bbox[3],
            ]
        )
        print('Predict box:', image_pd)
        f = open(
            root.format(
                n=idx + 1, m=str(layer1) + '_' + str(layer2) + '_' + str(layer3)
            ),
            'w',
        )
        f.write(','.join([str(i) for i in image_pd]))
        f.close()
        gt_list.append(image_gt)
        pd_list.append(image_pd)
        x, y, w, h = [int(round(t)) for t in image_pd]
        image_plot = cv2.rectangle(
            image_plot, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2
        )

        fig, ax = plt.subplots(1, 3, figsize=(20, 5))
        plt.ion()
        ax[0].imshow(template_plot)
        ax[1].imshow(image_plot)
        ax[2].imshow(similarity, 'jet')

        plt.savefig(
            root.format(
                n=idx + 1, m=str(layer1) + '_' + str(layer2) + '_' + str(layer3)
            )
            + '.png'
        )
        plt.pause(0.0001)
        plt.close()
        print('Done, results saved')
    return (gt_list, pd_list)


def get_parser():
    parser = argparse.ArgumentParser(description="DIT Feature Extractor")

    parser.add_argument(
        '--Dataset', default='BBS', help='specific a dataset to evaluate,BBS or KTM.'
    )
    parser.add_argument(
        '--Mode', default='Best', help='specific a mode to run,Best or All.'
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


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    dataset = args.Dataset
    file_dir = dataset + 'data'
    gt = sorted(
        [os.path.join(file_dir, i) for i in os.listdir(file_dir) if '.txt' in i]
    )
    img_path = sorted(
        [os.path.join(file_dir, i) for i in os.listdir(file_dir) if '.png' in i]
    )

    backbone_cfg = build_backbone_config(args)
    print(img_path)

    if args.Mode == 'All':
        layers = (2, 3, 4, 5, 6)
    else:
        if dataset == 'BBS':
            layers = (2, 3, 4)
        elif dataset == 'RMS':
            layers = (2, 3, 4)
        else:
            layers = (2, 3, 4)

    print(args.Dataset, args.Mode, layers)

    for i in range(len(layers)):
        for j in range(len(layers)):
            if i >= j:
                continue
            for k in range(len(layers)):
                if j >= k:
                    continue
                layer1 = layers[i]
                layer2 = layers[j]
                layer3 = layers[k]

                gt_list, pd_list = model_eval(
                    backbone_cfg, layer1, layer2, layer3, file_dir, True
                )

                iou_score = all_sample_iou(gt_list, pd_list)
                plot_success_curve(
                    iou_score,
                    file_dir
                    + '/'
                    + str(layer1)
                    + '_'
                    + str(layer2)
                    + '_'
                    + str(layer3),
                )
