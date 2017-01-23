__author__ = 'bsl'

import numpy as np

# label for train set
label_list_train = np.loadtxt('datasets/label_list_train1125_15182_x1.csv', delimiter=',')
print label_list_train.dtype
print label_list_train.shape
print label_list_train.__class__

aug_factor = 10
label_list_train_aug = np.vstack([label_list_train] * aug_factor)
print label_list_train_aug[1234, :] == label_list_train_aug[1234+label_list_train.shape[0]*5, :]
print label_list_train_aug[3456, :] == label_list_train_aug[3456+label_list_train.shape[0]*5, :]
print label_list_train_aug.shape

label_scalar = 1
label_list_train_aug_x1 = label_list_train_aug
np.savetxt('datasets/label_list_train1125_15182_aug{}_x{}.csv'.format(aug_factor, label_scalar), label_list_train_aug_x1, delimiter=',')
print label_list_train_aug_x1.max()

label_scalar = 10
label_list_train_aug_x10 = label_list_train_aug * 10
np.savetxt('datasets/label_list_train1125_15182_aug{}_x{}.csv'.format(aug_factor, label_scalar), label_list_train_aug_x10, delimiter=',')
print label_list_train_aug_x10.max()

# label for valid set, scaling only, no augmentation
label_list_valid = np.loadtxt('datasets/label_list_valid1215_2000_x1.csv', delimiter=',')
print label_list_valid.dtype
print label_list_valid.shape
print label_list_valid.__class__

label_scalar = 10
label_list_valid_x10 = label_list_valid*10
np.savetxt('datasets/label_list_valid1215_2000_x{}.csv'.format(label_scalar), label_list_valid_x10, delimiter=',')
print label_list_valid_x10.max()