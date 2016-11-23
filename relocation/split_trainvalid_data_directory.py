__author__ = 'bsl'

import numpy as np
from shutil import copyfile


# run this .py only once to generate train and test images and put them in diff directories
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

dirname_label = 'datasets/srcb_routeP1-3-10-14_label/'
label_filename = 'label_list.csv'
# load labels
image_filename_list, label_array = load_labels(dirname_label+label_filename)
nb_total_image_set = 75910
nb_test_image_set = 5000    # out of 75910 images, rest 70910 for the training
nb_train_image_set = nb_total_image_set - nb_test_image_set

rand_sample_index = np.random.choice(nb_total_image_set, nb_test_image_set, replace=False)
mask_for_test = np.zeros(nb_total_image_set, dtype=bool)
mask_for_test[rand_sample_index] = True

train_label_list = []
test_label_list = []
for i in np.arange(nb_total_image_set):
    if mask_for_test[i]:    # test image
        copyfile('datasets/srcb_routeP1-3-10-14_panoimage/'+image_filename_list[i],
                 'datasets/train_test_split/test/'+image_filename_list[i])
        test_label_list.append(label_array[i,:])
    else:                   # train image
        copyfile('datasets/srcb_routeP1-3-10-14_panoimage/' + image_filename_list[i],
                 'datasets/train_test_split/train/' + image_filename_list[i])
        train_label_list.append(label_array[i,:])
train_label_array = np.array(train_label_list)
test_label_array = np.array(test_label_list)
print 'train label shape: ' + str(train_label_array.shape)
print 'test label shape: ' + str(test_label_array.shape)

# save to csv file
np.savetxt('datasets/train_test_split/train_label.csv', train_label_array, delimiter=',')
np.savetxt('datasets/train_test_split/test_label.csv', test_label_array, delimiter=',')
