import numpy as np
import torch
import torch.nn.functional as F


def _l2_rec(src, trg):
    return torch.sum((src - trg)**2) / (src.shape[0] * src.shape[1])


def _ent(out):
    return - torch.mean(torch.log(F.softmax(out + 1e-6)))


def _discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))


def _ring(feat, type='geman'):
    x = feat.pow(2).sum(dim=1).pow(0.5)
    radius = x.mean()
    radius = radius.expand_as(x)
    # print(radius)
    if type == 'geman':
        l2_loss = (x - radius).pow(2).sum(dim=0) / (x.shape[0] * 0.5)
        return l2_loss
    else:
        raise NotImplementedError("Only 'geman' is implemented")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)

def dense_to_one_hot(labels_dense):
    """Convert class labels from scalars to one-hot vectors."""
    labels_one_hot = np.zeros((len(labels_dense),))
    labels_dense = list(labels_dense)
    for i, t in enumerate(labels_dense):
        if t == 10:
            t = 0
            labels_one_hot[i] = t
        else:
            labels_one_hot[i] = t
    return labels_one_hot
