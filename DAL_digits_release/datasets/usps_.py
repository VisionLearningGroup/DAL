import numpy as np
from scipy.io import loadmat
import gzip
import pickle
import sys
sys.path.append('../utils/')
from utils.utils import dense_to_one_hot
base_dir = './data'

def load_usps(all_use=False):
    dataset  = loadmat(base_dir + '/usps_28x28.mat')
    data_set = dataset['dataset']
    img_train = data_set[0][0]
    label_train = data_set[0][1]
    img_test = data_set[1][0]
    label_test = data_set[1][1]
    inds = np.random.permutation(img_train.shape[0])
    img_train = img_train[inds]
    label_train = label_train[inds]
    
    img_train = img_train * 255
    img_test = img_test * 255
    img_train = img_train.reshape((img_train.shape[0], 1, 28, 28))
    img_test = img_test.reshape((img_test.shape[0], 1, 28, 28))

    label_train = dense_to_one_hot(label_train)
    label_test = dense_to_one_hot(label_test)
    img_train = np.concatenate([ img_train, img_train, img_train], 1)
    img_test = np.concatenate([img_test, img_test,img_test],1)
    
    print('usps train X shape->',  img_train.shape)
    print('usps train y shape->',  label_train.shape)
    print('usps test X shape->',  img_test.shape)
    print('usps test y shape->', label_test.shape)


    return img_train, label_train, img_test, label_test
