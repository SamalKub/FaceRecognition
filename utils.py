import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import tqdm
import torch.nn.functional as F
import numpy as np
import time
import copy
import sklearn.metrics
from scipy.optimize import brentq
from scipy import interpolate

import torchvision.transforms as transforms

from torch.autograd import Variable
import os
from pathlib import Path
import glob
# from facenet_pytorch import InceptionResnetV1
import matplotlib.pyplot as plt
import collections
import PIL.Image
from torch.utils import data
from scipy.interpolate import interp1d

class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

def get_paths(dir_lfw: str, file_ext='.jpg'):
    path_list = []
    issame_list = 3000 * [1] + 3000 * [0]
    # images = iter(glob.iglob(dir_lfw+'/**/*'+file_ext,recursive = True))
    path_list = list(glob.iglob(dir_lfw+'/**/*'+file_ext, recursive=True))
    return path_list, issame_list

class LFWDataset(data.Dataset):
    '''
        Dataset subclass for loading LFW images in PyTorch.
        This returns multiple images in a batch.
    '''

    def __init__(self, path_list, issame_list, transforms, split = 'test'):
        '''
            Parameters
            ----------
            path_list    -   List of full path-names to LFW images
        '''
        self.files = collections.defaultdict(list)
        self.split = split
        self.files[split] =  path_list
        self.pair_label = issame_list
        self.transforms = transforms

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_file = self.files[self.split][index]
        img = PIL.Image.open(img_file)
        # print(img_file)
        im_out = self.transforms(img)
        return im_out

def imshow(inp, title=None, plt_ax=plt, default=False):
    """Imshow для тензоров"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt_ax.imshow(inp)
    if title is not None:
        plt_ax.set_title(title)
    plt_ax.grid(False)
def plot_roc_curve(fpr, tpr, roc_auc, fig_path=None):
    fig = plt.figure()
    plt.title('ROC - lfw dev-test')
    plt.plot(fpr, tpr, lw=2, label='ROC (auc = %0.4f)' % roc_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    # plt.show()
    # fig_path = '/content/gdrive/MyDrive/Samal_experiments/DL/FaceRecognition/plots'
    if fig_path:
        plt.savefig(fig_path, bbox_inches='tight')
    else:
        plt.show()

def plot_far_frr(far, frr, thresholds, eer, thresh):
    fig, ax = plt.subplots()

    ax.plot(thresholds, far, 'r--', label='FAR')
    ax.plot(thresholds, frr, 'g--', label='FRR')
    plt.xlabel('Threshold')
    plt.plot(thresh,eer,'ro', label='EER') 


    legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')

    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('C0')

    plt.show()
