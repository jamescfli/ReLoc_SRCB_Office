__author__ = 'bsl'

import numpy as np
from shutil import copyfile


# run this .py only once to generate train and test images (with label files)
# and put them in different folders
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
dirname_label = 'datasets/srcb_routeP1-3-10-14_480x1920/'   # 15182 + 1 label file
label_filename = 'label_list.csv'
# load labels
image_filename_list, label_array = load_labels(dirname_label+label_filename)
nb_total_image_set = 15182
nb_test_image_set = 2000    # out of 15182 images, rest 13182 for the training
nb_train_image_set = nb_total_image_set - nb_test_image_set

seed = 7    # same splitting random seed for verification
np.random.seed(seed)

rand_sample_index = np.random.choice(nb_total_image_set, nb_test_image_set, replace=False)
mask_for_test = np.zeros(nb_total_image_set, dtype=bool)    # all False
mask_for_test[rand_sample_index] = True                     # select test set index

train_label_list = []
test_label_list = []
for i in np.arange(nb_total_image_set):
    if mask_for_test[i]:    # test image
        copyfile('datasets/srcb_routeP1-3-10-14_480x1920/'+image_filename_list[i],
                 'datasets/train_test_split_480x1920/test/test_subdir'+image_filename_list[i])
        test_label_list.append(label_array[i,:])
    else:                   # train image
        copyfile('datasets/srcb_routeP1-3-10-14_480x1920/' + image_filename_list[i],
                 'datasets/train_test_split_480x1920/train/train_subdir' + image_filename_list[i])
        train_label_list.append(label_array[i,:])
train_label_array = np.array(train_label_list)
test_label_array = np.array(test_label_list)
print 'train label shape: ' + str(train_label_array.shape)
print 'test label shape: ' + str(test_label_array.shape)

# save to csv file
np.savetxt('datasets/train_test_split_480x1920/train_label.csv', train_label_array, delimiter=',')
np.savetxt('datasets/train_test_split_480x1920/test_label.csv', test_label_array, delimiter=',')
