import numpy as np
from scipy.io import loadmat
import sys

sys.path.append('../utils/')
from utils.utils import dense_to_one_hot

base_dir = './data'

def load_syn(scale=True, usps=False, all_use=False):
    syn_train = loadmat(base_dir + '/synth_train_28x28.mat')
    syn_test = loadmat(base_dir + '/synth_test_28x28.mat')
    syn_train_im = syn_train['X']
    syn_train_im = syn_train_im.transpose(3, 2, 0, 1).astype(np.float32)
    train_label = dense_to_one_hot(syn_train['y'])
    syn_test_im = syn_test['X']
    syn_test_im = syn_test_im.transpose(3, 2, 0, 1).astype(np.float32)
    test_label = dense_to_one_hot(syn_test['y'])
    print('syn number train X shape->',  syn_train_im.shape)
    print('syn number train y shape->',  train_label.shape)
    print('syn number test X shape->',  syn_test_im.shape)
    print('syn number test y shape->', test_label.shape)
    return syn_train_im, train_label, syn_test_im, test_label



