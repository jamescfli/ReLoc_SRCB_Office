__author__ = 'bsl'

import numpy as np
from shutil import copyfile


# split 2000 img testing set from 20161215 folder
def load_labels(label_filename):
    with open(label_filename, 'r') as label_file:
        lines = label_file.readlines()
    image_filename_list = []
    nb_total_sample = lines.__len__()
    label_array = np.zeros((nb_total_sample, 2), dtype='float32')
    for index, line in enumerate(lines):
        items = line.split(',')
        image_filename_list.append(items[0])
        label_array[index, :] = np.array(items[1:])     # cut the image name
    return image_filename_list, label_array

# data without augmentation
dirname_label = 'datasets/test_image_20161215/'
label_filename = 'label_list_480x1920_x1_wfilename.csv'
# load labels
image_filename_list, label_array = load_labels(dirname_label+label_filename)
nb_total_image_set = 12526
nb_test_image_set = 2000

seed = 7
np.random.seed(seed)

rand_sample_index = np.random.choice(nb_total_image_set, nb_test_image_set, replace=False)
mask_for_test = np.zeros(nb_total_image_set, dtype=bool)    # all False
mask_for_test[rand_sample_index] = True                     # select test set index

test_label_list = []
for i in np.arange(nb_total_image_set):
    if mask_for_test[i]:    # test image
        copyfile('datasets/test_image_20161215/image_480x1920/image_480x1920_subdir/'+image_filename_list[i],
                 'datasets/test_image_20161215/image_480x1920_2000_for_test/image_480x1920_2000/'+image_filename_list[i])
        test_label_list.append(label_array[i,:])
test_label_array = np.array(test_label_list)
print 'test label shape: {}'.format(test_label_array.shape)
# upscale by 100x
test_label_array = 100 * test_label_array

# save to csv file
np.savetxt('datasets/test_image_20161215/label_list_480x1920_2000_x100.csv', test_label_array, delimiter=',')