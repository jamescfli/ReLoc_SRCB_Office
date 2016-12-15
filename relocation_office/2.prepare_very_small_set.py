__author__ = 'bsl'

import numpy as np
import os
from shutil import copyfile

# prepare small set with 20 images from testing set
nb_sample = 20
candidate_dir = 'datasets/train_test_split_480x1920_20161125/test/test_subdir/'
candidate_namelist = sorted(os.listdir(candidate_dir))      # reverse = False
# read x100 label
candidate_label = np.loadtxt('datasets/train_test_split_480x1920_20161125/test_label_x100.csv', dtype='float32', delimiter=',')
assert candidate_label.shape[0] == candidate_namelist.__len__(), 'file number != label number'
# selected
selected_namelist = list(candidate_namelist[i] for i in np.arange(nb_sample)*100)
selected_label = candidate_label[np.arange(nb_sample)*100, :]
# cp selected images to '20_image_set' directory
for i in np.arange(nb_sample):
    copyfile('datasets/train_test_split_480x1920_20161125/test/test_subdir/'+selected_namelist[i],
             'datasets/train_test_split_480x1920_20161125/20_image_set/20images/'+selected_namelist[i])
# save label to corresponding .csv file
np.savetxt('datasets/train_test_split_480x1920_20161125/20imageset_label_x100.csv', selected_label, delimiter=',')
