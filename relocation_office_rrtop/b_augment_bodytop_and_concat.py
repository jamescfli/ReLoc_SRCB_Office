import csv
from PIL import Image as pil_image
from utils.convert_equirec_to_cube import convert_back_array_wrapper, cut_body_part, cut_top_face
from utils.equirec_img_augmentation import random_vshift, random_rotation
import numpy as np

augmentation_factor = 10    # 5 may give less options for top face rotation, so take 10
vshift_max_range = 0.1      # 10% up and down shift
rotation_limit_in_degree = 180      # clockwise and anti-clockwise for 180 degrees

# take care of training set first
with open('datasets/label_list_w_filename_train1125_15182_x1.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        whole_img_in_array = np.array(pil_image.open('datasets/train_960x1920_20161125/whole/'+row[0]), dtype='uint8')
        body_img_in_array = cut_body_part(whole_img_in_array)
        cube_img_in_array = convert_back_array_wrapper(whole_img_in_array)
        top_img_in_array = cut_top_face(cube_img_in_array)
        # back up body and top before augmentation
        pil_image.fromarray(body_img_in_array).save('datasets/train_960x1920_20161125/body/body_'+row[0])
        pil_image.fromarray(top_img_in_array).save('datasets/train_960x1920_20161125/top/top_'+row[0])

        # augmentation
        for i in np.arange(augmentation_factor):
            aug_body_img_in_array = random_vshift(img_src_in_array=body_img_in_array,
                                                  vshift_range_limit=vshift_max_range)
            aug_top_img_in_array = random_rotation(img_src_in_array=top_img_in_array,
                                                   rg=rotation_limit_in_degree)
            aug_concat_in_array = np.hstack((aug_body_img_in_array, aug_top_img_in_array))
            # save to folder with index
            pil_image.fromarray(aug_concat_in_array).save('datasets/train_960x1920_20161125/'
                                                          'aug_10_times_body_top_concat/concat_{}_{}'
                                                          .format(i, row[0]))

# no need to augment validation set, but concatenation is necessary
