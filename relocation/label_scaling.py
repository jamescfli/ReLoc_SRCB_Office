__author__ = 'bsl'

import numpy as np


train_label = np.loadtxt('datasets/train_test_split_480x1920/train_label_x1.csv', dtype='float32', delimiter=',')
test_label = np.loadtxt('datasets/train_test_split_480x1920/test_label_x1.csv', dtype='float32', delimiter=',')
# .. train_label[:, 0].min() = 0, .max() = 0.89297998, [:, 1].min() = 0.15278, .max() = 0.92799006
# .. test_label[:, 0].min() = 0, .max() = 0.89297003, [:, 1].min() = 0.15278, .max() = 0.92796999

train_label = 100*train_label
test_label = 100*test_label

# save labels in new files
np.savetxt('datasets/train_test_split_480x1920/train_label_x100.csv', train_label, delimiter=',')
np.savetxt('datasets/train_test_split_480x1920/test_label_x100.csv', test_label, delimiter=',')
