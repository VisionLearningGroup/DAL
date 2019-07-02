from scipy.io import loadmat
import numpy as np
import sys

sys.path.append('./utils/')
from utils.utils import dense_to_one_hot
from scipy.misc import imresize
base_dir = './data'
def load_svhn():
    svhn_train = loadmat(base_dir + '/svhn_train_28x28.mat')
    svhn_test = loadmat(base_dir + '/svhn_test_28x28.mat')
    svhn_train_im = svhn_train['X']
    svhn_train_im = svhn_train_im.transpose(3, 2, 0, 1).astype(np.float32)
    svhn_label = dense_to_one_hot(svhn_train['y'])
    svhn_test_im = svhn_test['X']
    svhn_test_im = svhn_test_im.transpose(3, 2, 0, 1).astype(np.float32)
    svhn_label_test = dense_to_one_hot(svhn_test['y'])
    print('svhn train X shape->',  svhn_train_im.shape)
    print('svhn train y shape->',  svhn_label.shape)
    print('svhn test X shape->',  svhn_test_im.shape)
    print('svhn test y shape->', svhn_label_test.shape)

    return svhn_train_im, svhn_label, svhn_test_im, svhn_label_test
