import sys
import numpy as np
sys.path.append('../loader')
from unaligned_data_loader import UnalignedDataLoader
from svhn import load_svhn
from mnist import load_mnist
from mnist_m import load_mnistm
from usps_ import load_usps
from synth_number import load_syn


def return_dataset(data, scale=False, usps=False, all_use='no'):
    if data == 'svhn':
        train_image, train_label, test_image, test_label = load_svhn()
    if data == 'mnist':
        train_image, train_label, test_image, test_label = load_mnist(scale=scale, usps=usps, all_use=all_use)
        print(train_image.shape)
    if data == 'usps':
        train_image, train_label, test_image, test_label = load_usps(all_use=all_use)
    if data == 'mnistm':
        train_image, train_label, test_image, test_label = load_mnistm()
    if data == 'synth':
        train_image, train_label, test_image, test_label = load_syntraffic()
    if data == 'gtsrb':
        train_image, train_label, test_image, test_label = load_gtsrb()
    if data == 'syn':
        train_image, train_label, test_image, test_label = load_syn()

    return train_image, train_label, test_image, test_label


def dataset_read(source, target, batch_size, scale=False, all_use='no'):
    S = {}
    S_test = {}
    T = {}
    T_test = {}
    usps = False
    if source == 'usps' or target == 'usps':
        usps = True

    domain_all = ['mnistm', 'mnist', 'usps', 'svhn', 'syn']
    domain_all.remove(source)
    train_source, s_label_train, test_source, s_label_test = return_dataset(source, scale=scale,usps=usps, all_use=all_use)

    train_target, t_label_train, test_target, t_label_test = return_dataset(domain_all[0], scale=scale, usps=usps,
                                                                                all_use=all_use)
    for i in range(1, len(domain_all)):
        train_target_, t_label_train_, test_target_, t_label_test_ = return_dataset(domain_all[i], scale=scale, usps=usps, all_use=all_use)
        train_target = np.concatenate((train_target, train_target_), axis=0)
        t_label_train = np.concatenate((t_label_train, t_label_train_), axis=0)
        test_target = np.concatenate((test_target, test_target_), axis=0)
        t_label_test = np.concatenate((t_label_test, t_label_test_), axis=0)

    # print(domain)
    print('Source Training: ', train_source.shape)
    print('Source Training label: ', s_label_train.shape)
    print('Source Test: ', test_source.shape)
    print('Source Test label: ', s_label_test.shape)

    print('Target Training: ', train_target.shape)
    print('Target Training label: ', t_label_train.shape)
    print('Target Test: ', test_target.shape)
    print('Target Test label: ', t_label_test.shape)




    S['imgs'] = train_source
    S['labels'] = s_label_train
    T['imgs'] = train_target
    T['labels'] = t_label_train

    # input target samples for both 
    S_test['imgs'] = test_target
    S_test['labels'] = t_label_test
    T_test['imgs'] = test_target
    T_test['labels'] = t_label_test
    scale = 32 if source == 'synth' else 32 if source == 'usps' or target == 'usps' else 32
    train_loader = UnalignedDataLoader()
    train_loader.initialize(S, T, batch_size, batch_size, scale=scale)
    dataset = train_loader.load_data()
    test_loader = UnalignedDataLoader()
    test_loader.initialize(S_test, T_test, batch_size, batch_size, scale=scale)
    dataset_test = test_loader.load_data()
    return dataset, dataset_test
