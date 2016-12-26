__author__ = 'bsl'

# from keras.preprocessing.image import ImageDataGenerator
from utils.custom_image import ImageDataGenerator
import numpy as np

if __name__ == '__main__':
    img_height = 480
    img_width = img_height*4

    # limit the range by considering distorition and fingers
    height_shift_ratio = 0.1    # original: 0.25 to 0.75 +/- 10% = 0.15 to 0.85
    batch_size = 32
    # batch_size = 3

    data_gen_aug = ImageDataGenerator(rescale=1. / 255,
                                      height_shift_range=height_shift_ratio)
    generator_aug = data_gen_aug.flow_from_directory(
        # 'datasets/test_image_20161215/image_960x1920/image_960x1920_subdir/',   # 12526 imgs
        # 'datasets/test_image_20161215/image_960x1920/img_960x1920_test3img_subdir/',   # 3 imgs
        "datasets/train_test_split_480x1920_20161125/train/train_subdir/",      # 13182 imgs
        target_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=False,
        save_to_dir='datasets/train_test_split_480x1920_20161125/train_augmented/train_augmented_subdir/',
        save_prefix='aug_img',
        class_mode='xy_pos',
        # label_file="../../label_list_960x1920_x1.csv")
        # label_file="../../label_list_960x1920_test3img_x1.csv")
        label_file="../../train_label_x100.csv")

    # x_batch, y_batch = generator_aug.next()
    # print y_batch

    nb_gen_sample = 13182

    for i in np.arange(nb_gen_sample/batch_size):   # up to 411, 0~410
        x_batch, y_batch = generator_aug.next()
        print i
        print y_batch
    x_batch, y_batch = generator_aug.next()
    print 'last {}'.format(nb_gen_sample/batch_size)    # 411
    print y_batch
