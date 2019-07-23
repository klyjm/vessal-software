import json
import os
# import torch
from torch import load as torch_load
from  torch import cuda as torch_cuda
from torch import device as torch_device
from torchvision import transforms
from PIL import Image
from model import vgg11_bn
# import numpy as np


class ImageTransforms(object):

    def __init__(self, name, size, normalize):
        self.transfs = {
            'val': transforms.Compose([
                transforms.CenterCrop(size=size),
                transforms.Resize(size=128, interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([normalize[0]], [normalize[1]])
            ]),
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=90),
                transforms.CenterCrop(size=size),
                transforms.Resize(size=128, interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([normalize[0]], [normalize[1]])
            ])
        }[name]

    def apply(self, data, target):
        return self.transfs(data), target


def _get_transform(config):
    tsf_size = config['transforms']['args']['size']
    tsf_normalize = config['transforms']['args']['normalize']
    return ImageTransforms('val', tsf_size, tsf_normalize)


def _get_model_att(checkpoint):
    m_name = checkpoint['config']['model']['type']
    sd = checkpoint['state_dict']
    classes = checkpoint['classes']
    return m_name, sd, classes


class ImageClassifier:
    def __init__(self, path):
        self.config_path = os.path.join(path, 'config.json')
        self.model_path = os.path.join(path, 'model_best.pth')

        config = json.load(open(self.config_path))
        checkpoint = torch_load(self.model_path, map_location='cpu')
        m_name, sd, self.classes = _get_model_att(checkpoint)

        model = vgg11_bn(self.classes, pretrained=False)
        model.load_state_dict(checkpoint['state_dict'])

        tsf = _get_transform(config)

        # self.device = torch_device("cuda" if torch_cuda.is_available() else "cpu")
        self.device = torch_device("cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.transforms = tsf

    def infer(self, img_list):
        label_all = []
        conf_all = []

        for image_np in img_list:
            image = Image.fromarray(image_np).convert('RGB')
            image_t, _ = self.transforms.apply(image, None)
            label, conf = self.model.predict(image_t.to(self.device))
            label_all.append(label)
            conf_all.append(conf)

        return label_all, conf_all


# if __name__ == '__main__':
#     model_root = 'merge12A_0628_001439'
#     classifier = ImageClassifier(model_root)
#
#     image_root = 'merge12A_0628_001439/image'
#     image_name_list = os.listdir(image_root)
#     image_list = []
#     for image_name in image_name_list:
#         image_list.append(np.array(Image.open(os.path.join(image_root, image_name)).convert('RGB')))
#
#     label_all, conf_all = classifier.infer(image_list)
#     print(1)
