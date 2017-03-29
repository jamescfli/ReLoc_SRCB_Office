__author__ = 'bsl'

from relocation_office_rrtop.d_build_parallel_model_bodytop import build_2path_vgg_bodytopf_model
from utils.custom_image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from utils.timer import Timer


def generate():
    img_height = 448
    initial_weights = 'imagenet'
    nb_hidden_node = 2048  # where fc layer for topf will be divided by 4, i.e. 512
    learning_rate = 1e-3  # to conv layers
    lr_multiplier = 1.0  # to top fc layers
    l1_regular = 0.0  # weight decay in L1 norm
    l2_regular = 1.e+0  # L2 norm
    label_scalar = 1  # expend from [0, 1]
    flag_add_bn = True
    flag_add_do = False
    # do_ratio = 0.5
    batch_size = 8  # tried 32 (224), 3850MB
    nb_epoch = 2  # due to higher dimension of 448 img @ network bottle-neck
    nb_epoch_annealing = 1  # anneal for every <> epochs
    annealing_factor = 0.1
    np.random.seed(7)  # to repeat results
    model_stacked = build_2path_vgg_bodytopf_model(img_height=img_height,
                                                   weights=initial_weights,
                                                   nb_fc_hidden_node=nb_hidden_node,
                                                   global_learning_rate=learning_rate,
                                                   learning_rate_multiplier=lr_multiplier,
                                                   l1_regularization=l1_regular,
                                                   l2_regularization=l2_regular,
                                                   is_bn_enabled=flag_add_bn,
                                                   is_do_enabled=flag_add_do)
    model_path = 'models/'
    weight_filename = 'weights_input448_fc2048bodytopf_imagenet_1125imgx10_ls1_2epoch_sgdlr1e-3m1ae1af01_l1reg0l2reg1_reloc_model.h5'
    model_stacked.load_weights(model_path + weight_filename, by_name=True)
    model_stacked.summary()

    img_width = img_height * 5
    batch_size = 1
    nb_valid_sample = 2000

    datagen_valid = ImageDataGenerator(rescale=1. / 255)
    generator_valid = datagen_valid.flow_from_directory('datasets/valid_480x2400_concat_nb2000_20161215/concat/',
                                                        target_size=(img_height, img_width),
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        class_mode='xy_pos',
                                                        label_file="../../label_list_valid1215_2000_x{}.csv"
                                                        .format(label_scalar))
    with Timer('Generate validation position xy'):
        valid_pos = model_stacked.predict_generator(generator_valid,
                                                    val_samples=nb_valid_sample)
    # 225sec for 2000 448 img

    print "valid pos xy shape: {}".format(valid_pos.shape)
    np.save(open('predicted_data/valid_position_result_w20161215img.npy', 'w'), valid_pos)


def visualize():
    valid_xy = np.load(open('predicted_data/valid_position_result_w20161215img.npy'))
    valid_pos_gt = np.loadtxt('datasets/label_list_valid1215_2000_x1.csv',
                              dtype='float32', delimiter=',')
    # compare
    plt.plot(np.arange(2000) + 1, valid_pos_gt[:, 0], 'b')
    plt.plot(np.arange(2000) + 1, valid_pos_gt[:, 1], 'r')
    plt.plot(np.arange(2000) + 1, valid_xy[:, 0], 'c')
    plt.plot(np.arange(2000) + 1, valid_xy[:, 1], 'orange')
    plt.legend(['x_gt', 'y_gt', 'x_valid', 'y_valid'], loc='lower right')
    plt.show()

if __name__ == '__main__':
    # generate()
    visualize()



