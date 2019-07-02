from scipy.io import loadmat
import scipy.misc
import numpy as np
import sys


data_set = loadmat('usps_28x28.mat')
dataset = data_set['dataset']

img = dataset[0][0]

print(img.shape)
img = img[0][0]
img = img*255  
print(img.shape)
scipy.misc.toimage(img,cmin=0.0, cmax=255).save('usps.jpg')


svhn_train = loadmat('train_32x32.mat')
svhn_train_im = svhn_train['X']
