import torch
import torchvision
from torchvision import datasets, transforms
import PIL

# https://github.com/pytorch/vision/issues/3497
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)
#######################################################

TORCH_DATASET_DIR = '/home/rjia/datasets/torch_datasets/'


mnist = datasets.MNIST(root=TORCH_DATASET_DIR, download=True)

