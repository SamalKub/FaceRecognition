import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
import numpy as np
from utils import CenterLoss, plot_far_frr, plot_roc_curve
import tqdm
import copy
from torch.autograd import Variable
import sklearn.metrics
from scipy.optimize import brentq
from scipy import interpolate
import torch.optim as optim
from torch.optim import lr_scheduler
from scipy.interpolate import interp1d


def fit_epoch(model, train_loader, criterion, optimizer, name, epoch, num_epoch, device, alpha=0.02, flag_center=False, criterion_centerloss=None, optimizer_center=None):
    running_loss = 0.0
    running_corrects = 0

    # Set model to training mode
    model.train()

    # Iterate over data.
    for step, (inputs, labels) in enumerate(train_loader):
        # print('Batch {}/{}'.format(step, len(train_loader) - 1)) 
        print('Epoch {}/{} Batch {}/{}'.format(epoch, num_epoch - 1, step, len(train_loader) - 1))

        inputs = Variable(inputs).to(device)
        labels = Variable(labels).to(device)

        # Forward
        # Get model outputs and calculate loss
        feat_center, outputs = model(inputs)
        loss_softmax = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        if flag_center:
            loss_center = criterion_centerloss(feat_center, labels)
            loss = loss_softmax + loss_center * alpha #multi-task loss
            # Zero the parameter gradients
            optimizer.zero_grad()
            optimizer_center.zero_grad()
        else:
            loss = loss_softmax
            optimizer.zero_grad()

        loss.backward()

        acc = torch.sum(preds == labels.data).double() / inputs.data.size(0)
        if flag_center:
            # multiple (1./alpha) in order to remove the effect of alpha on updating centers
            for param in criterion_centerloss.parameters():
                param.grad.data *= (1. / alpha)
            optimizer.step()
            optimizer_center.step()
            print(
            'Train Batch Loss: {:.4f} Train Batch Softmax Loss: {:.4f} Train Batch Center Loss: {:.4f} Acc: {:.4f}'.format(
                        loss.data.item(), loss_softmax.data.item(), loss_center.data.item() * alpha,
                        acc))
        else:
            optimizer.step()
            print(
            'Train Batch Loss: {:.4f} Acc: {:.4f}'.format(
                    loss.data.item(),
                    acc))

        # Statistics
        running_loss += loss.data.item() * inputs.data.size(0)
        running_corrects += torch.sum(preds == labels.data)
    return loss, acc, running_loss, running_corrects

def eval_epoch(model, val_loader, criterion, name, epoch, issame_list, device, distance='euclidean', fig_path=None):
    # Set model to evaluate mode
    model.eval()
    # Feature extraction
    features = []
    for batch_idx, images in tqdm.tqdm(enumerate(val_loader), total=len(val_loader), desc='Extracting features'):
        x = Variable(images).to(device)
        feat, _ = model(x)
        feat = feat.data.cpu()
        features.append(feat)
    features = torch.stack(features)
    features = F.normalize(features, p=2, dim=1)  # L2-normalize

    # Verification
    num_feat = features.size()[0]
    feat_pair1 = features[np.arange(0, num_feat, 2), :]
    feat_pair2 = features[np.arange(1, num_feat, 2), :]
    if distance=='euclidean':
        feat_dist = (feat_pair1 - feat_pair2).norm(p=2, dim=1)
    else:
        feat_dist = 1 - F.cosine_similarity(feat_pair1, feat_pair2, -1)
    feat_dist = feat_dist.numpy()

    # Eval metrics
    scores = -feat_dist
    gt = np.asarray(issame_list)

    roc_auc = sklearn.metrics.roc_auc_score(gt, scores)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(gt, scores)
    print('ROC-AUC: %.04f' % roc_auc)
    plot_roc_curve(fpr, tpr, roc_auc, fig_path+'/{}Epoch{}.png'.format(name, epoch))

    roc_eer = brentq(
                lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)


    print(('LFW VAL AUC: %0.4f, LFW VAL EER: %0.4f') % (roc_auc, roc_eer))
    return roc_auc, roc_eer

def train(train_loader, val_loader, model, optimizer, criterion, issame_list, device, scheduler=None, num_epochs=10,
          distance='euclidean', fig_path=None,
          flag_center=False, criterion_centerloss=None, optimizer_center=None,
          logs_save=None, checkpoint_save=None, model_name=None, alpha=0.02):
    train_batch_loss_history = []
    train_batch_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_roc_auc = 0.0
    fig_path = fig_path
    path_save = logs_save
    path_models = checkpoint_save

    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        loss, acc, running_loss, running_corrects = fit_epoch(model, train_loader,
                                                              criterion, optimizer, model_name, epoch, num_epochs, device, flag_center=flag_center, criterion_centerloss=criterion_centerloss, optimizer_center=optimizer_center, alpha=alpha)
        
        train_batch_loss_history.append(loss.data.item())
        train_batch_acc_history.append(acc)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print('Train Epoch Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        epoch_val_roc_auc, epoch_val_roc_eer = eval_epoch(model, val_loader, criterion, model_name, epoch, issame_list, device, distance=distance,fig_path=fig_path)
        
        if scheduler is not None:
            scheduler.step()
        if epoch_val_roc_auc > best_roc_auc:
            best_roc_auc = epoch_val_roc_auc
            best_model_wts = copy.deepcopy(model.state_dict())

            if flag_center:
                # Save checkpoint
                torch.save({
                    'epoch': epoch,
                    'arch': model.__class__.__name__,
                    'optim_softmax_state_dict': optimizer.state_dict(),
                    'optim_center_state_dict': optimizer_center.state_dict(),
                    'model_state_dict': model.state_dict(),
                    'train_epoch_loss': epoch_loss,
                    'train_epoch_acc': epoch_acc,
                    'best_roc_auc': best_roc_auc,
                }, path_models+'/{}_CASIA-WEB-FACE-Aligned_Epoch_{}_LfwAUC_{}.tar'.format(model_name, epoch, best_roc_auc))
            else:
                # Save checkpoint
                torch.save({
                    'epoch': epoch,
                    'arch': model.__class__.__name__,
                    'optim_softmax_state_dict': optimizer.state_dict(),
                    'model_state_dict': model.state_dict(),
                    'train_epoch_loss': epoch_loss,
                    'train_epoch_acc': epoch_acc,
                    'best_roc_auc': best_roc_auc,
                }, path_models+'/{}_CASIA-WEB-FACE-Aligned_Epoch_{}_LfwAUC_{}.tar'.format(model_name, epoch, best_roc_auc))
        print('Current Best val ROC AUC: {:4f}'.format(best_roc_auc))

    print('Training complete')
    # Save train_batch_loss_history and train_batch_acc_history
    with open(path_save+'/{}_train_batch_loss_history_Aligned.txt'.format(model_name), 'w') as f:
        for item in train_batch_loss_history:
            f.write("%s\n" % item)
    with open(path_save+'/{}_train_batch_acc_history_Aligned.txt'.format(model_name), 'w') as f:
        for item in train_batch_acc_history:
            f.write("%s\n" % item)
#     return train_batch_loss_history, train_batch_acc_history

def evaluate(model, optimizer, checkpoint_path, device, val_loader, issame_list,flag_center=False):
    model = model
    optimizer = optimizer

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optim_softmax_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['train_epoch_loss']
    if flag_center:
        optimizer_center.load_state_dict(checkpoint['optim_center_state_dict'])

    model.eval()
    # Feature extraction
    features = []
    for batch_idx, images in tqdm.tqdm(enumerate(val_loader), total=len(val_loader), desc='Extracting features'):
        x = Variable(images).to(device) # Test-time memory conservation
        feat, _ = model(x)
        feat = feat.data.cpu()
        # print('feat:', feat.shape)
        features.append(feat)
    features = torch.stack(features)
    features = F.normalize(features, p=2, dim=1)  # L2-normalize
    # Verification
    num_feat = features.size()[0]
    feat_pair1 = features[np.arange(0, num_feat, 2), :]
    feat_pair2 = features[np.arange(1, num_feat, 2), :]
    feat_dist = (feat_pair1 - feat_pair2).norm(p=2, dim=1)
    feat_dist = feat_dist.numpy()

    # Eval metrics
    scores = -feat_dist
    gt = np.asarray(issame_list)

    roc_auc = sklearn.metrics.roc_auc_score(gt, scores)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(gt, scores)
    print('ROC-AUC: %.04f' % roc_auc)

    plot_roc_curve(fpr, tpr, roc_auc)
 
    frr = 1 - tpr
    far = fpr

    roc_eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(roc_eer) 
    plot_far_frr(far, frr, thresholds, roc_eer, thresh)

    print(('LFW VAL AUC: %0.4f, LFW VAL EER: %0.4f') % (roc_auc, roc_eer))
    # print('FAR {}/FRR {}'.format(far, frr))
#     return fpr, tpr, thresholds, roc_auc, roc_eer

