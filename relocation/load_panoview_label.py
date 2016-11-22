__author__ = 'bsl'

import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.applications import vgg16


# TODO remember to let testing data to go through the same preprocessing
# e.g. dim switch, median reduction
def load_data():
    dirname_image = 'datasets/srcb_routeP1-3-10-14_panoimage/'      # 75912 images
    dirname_label = 'datasets/srcb_routeP1-3-10-14_label/'
    label_filename = 'label_list.csv'
    # load labels
    image_filename_list, label_array = load_labels(dirname_label+label_filename)
    image_array = np.array([img_to_array(Image.open(dirname_image+fname)) for fname in image_filename_list])
    # note vgg preprocess reuse applications.image_utils' preprocess_input
    return vgg16.preprocess_input(image_array), label_array


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


if __name__ == '__main__':
    image_array_result, label_array_result = load_data()
    print image_array_result.shape
    print label_array_result.shape