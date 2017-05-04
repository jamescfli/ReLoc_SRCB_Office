__author__ = 'bsl'

import csv
from PIL import Image as pil_image

import sys
sys.path.append('../')
from utils.convert_equirec_to_cube import convert_back_array_wrapper, cut_body_part, cut_top_face, cut_top_3_quarter

import numpy as np


# no need to augment validation set, but concatenation is necessary
with open('datasets/label_list_w_filename_valid1215_12526_x1.csv') as csvfile:
    nb_img = 12526       # 12526 imgs in valid1215 set
    nb_valid_img = 2000  # random choose 2000 imgs as the validation set
    label_list = []
    np.random.seed(7)
    img_choice_index = np.random.choice(np.arange(nb_img), size=nb_valid_img, replace=False)
    # back up chosen index
    np.savetxt('datasets/valid_nb2000_20161215/index_2000_valid.txt', img_choice_index, fmt='%d', delimiter=',')
    img_choice_mask = np.zeros(nb_img, dtype='bool')
    img_choice_mask[img_choice_index] = True
    # read the csv file
    reader = csv.reader(csvfile, delimiter=',')
    for i, row in enumerate(reader):
        if img_choice_mask[i]:
            # save concatenated img to validation folder
            whole_img_in_array = np.array(pil_image.open('datasets/valid_960x1920_20161215/whole/'+row[0]), dtype='uint8')
            body_img_in_array = cut_body_part(whole_img_in_array)
            cube_img_in_array = convert_back_array_wrapper(whole_img_in_array)
            top_img_in_array = cut_top_face(cube_img_in_array)
            # concat without vshift or rotation
            concat_in_array = np.hstack((body_img_in_array, top_img_in_array))
            pil_image.fromarray(concat_in_array)\
                .save('datasets/valid_nb2000_20161215/concat/concat_'+row[0])
            # cut and save top 3 quarters
            top_3_quarters_in_array = cut_top_3_quarter(whole_img_in_array)
            pil_image.fromarray(top_3_quarters_in_array)\
                .save('datasets/valid_nb2000_20161215/top3quarter/top3quarter_'+row[0])
            # save label to array
            label_list.append([row[1], row[2]])
    # convert with dtype, o.w. TypeError: Mismatch between array dtype ('|S7') and format specifier ('%.18e,%.18e')
    label_list_array = np.array(label_list, dtype='float32')
    assert nb_valid_img == label_list_array.shape[0], 'wrong nb of img for small valid set'
    np.savetxt('datasets/label_list_valid1215_2000_x1.csv', label_list_array, delimiter=',')

# import os
# 
# # draft the label file, by accident overwrite valid2000 folder
# with open('datasets/label_list_w_filename_valid1215_12526_x1.csv') as csvfile:
#     nb_img = 12526       # 12526 imgs in valid1215 set
#     nb_valid_img = 2000  # random choose 2000 imgs as the validation set
#     label_list = []
#     name_list_selected = sorted(os.listdir('datasets/valid_480x2400_concat_nb2000_20161215/concat/'))
#     index_selected = 0
#     # read the csv file
#     reader = csv.reader(csvfile, delimiter=',')
#     for i, row in enumerate(reader):
#         if row[0] == name_list_selected[index_selected][7:]:
#             label_list.append([row[1], row[2]])
#             index_selected += 1
#             if index_selected == nb_valid_img:
#                 # got all 2000 label already
#                 break
#     label_list_array = np.array(label_list, dtype='float32')
#     assert nb_valid_img == label_list_array.shape[0], 'wrong nb of img for small valid set'
#     np.savetxt('datasets/label_list_valid1215_2000_x1.csv', label_list_array, delimiter=',')
#     print label_list_array.max()
# 
#     label_list_array_x10 = label_list_array*10
#     np.savetxt('datasets/label_list_valid1215_2000_x10.csv', label_list_array_x10, delimiter=',')
#     print label_list_array_x10.max()

