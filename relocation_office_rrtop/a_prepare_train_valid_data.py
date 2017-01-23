__author__ = 'bsl'

# run with OpenCV
from utils.convert_equirec_to_cube import convert_back_cv2_wrapper, cut_top_face
import os
import numpy as np
import csv
import cv2

# # discard the file name in the label file
# filename_list = []
# position_list = []
# with open('datasets/label_list_w_filename_train1125_15182_x1.csv') as csvfile:
#     reader = csv.reader(csvfile, delimiter=',')
#     for row in reader:
#         filename_list.append(row[0])
#         position_list.append((row[1], row[2]))
# np.savetxt('datasets/label_list_train1125_15182_x1.csv', np.array(position_list, dtype='float32'), delimiter=',')
# filename_list = []
# position_list = []
# with open('datasets/label_list_w_filename_valid1215_12526_x1.csv') as csvfile:
#     reader = csv.reader(csvfile, delimiter=',')
#     for row in reader:
#         filename_list.append(row[0])
#         position_list.append((row[1], row[2]))
# np.savetxt('datasets/label_list_valid1215_12526_x1.csv', np.array(position_list, dtype='float32'), delimiter=',')

# convert whole pic to body part and top face
with open('datasets/label_list_w_filename_train1125_15182_x1.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        whole_img = cv2.imread('datasets/train_960x1920_20161125/whole/'+row[0], cv2.IMREAD_COLOR)
        src_img_height = whole_img.shape[0]
        body_img = whole_img[(src_img_height/4):(src_img_height*3/4), :, :]
        cube_img = convert_back_cv2_wrapper(whole_img)
        top_img = cut_top_face(cube_img)
        cv2.imwrite('datasets/train_960x1920_20161125/body/body_'+row[0], body_img)
        cv2.imwrite('datasets/train_960x1920_20161125/top/top_'+row[0], top_img)
with open('datasets/label_list_w_filename_valid1215_12526_x1.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        whole_img = cv2.imread('datasets/valid_960x1920_20161215/whole/' + row[0], cv2.IMREAD_COLOR)
        src_img_height = whole_img.shape[0]
        body_img = whole_img[(src_img_height / 4):(src_img_height * 3 / 4), :, :]
        cube_img = convert_back_cv2_wrapper(whole_img)
        top_img = cut_top_face(cube_img)
        cv2.imwrite('datasets/valid_960x1920_20161215/body/body_' + row[0], body_img)
        cv2.imwrite('datasets/valid_960x1920_20161215/top/top_' + row[0], top_img)
