__author__ = 'bsl'

import csv
from PIL import Image as pil_image
import numpy as np


with open('datasets/label_list_w_filename_train1125_15182_x1.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        body_img_in_array = np.array(pil_image.open('datasets/train_960x1920_20161125/body/body_'+row[0]), dtype='uint8')
        topf_img_in_array = np.array(pil_image.open('datasets/train_960x1920_20161125/top/top_'+row[0]), dtype='uint8')
        concat_img_in_array = np.hstack((body_img_in_array, topf_img_in_array))
        pil_image.fromarray(concat_img_in_array).save('datasets/train_960x1920_20161125/concat/concat_{}'.format(row[0]))
