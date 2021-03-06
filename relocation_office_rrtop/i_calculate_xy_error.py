__author__ = 'bsl'

from relocation_office_rrtop.d_build_singlepath_model_bodyonly import build_1path_vgg_bodyonly_model
from relocation_office_rrtop.d_build_parallel_model_bodytop import build_2path_vgg_bodytopf_model
from utils.custom_image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from utils.timer import Timer


def generate(train_valid_split = None, nb_path_for_model = None):
    img_height = 448
    initial_weights = 'imagenet'
    nb_hidden_dense_layer = 2   # nb of hidden fc layers, output dense excluded
    nb_hidden_node = 2048       # where fc layer for topf will be divided by 4, i.e. 512
    learning_rate = 1e-3        # to conv layers
    lr_multiplier = 1.0         # to top fc layers
    w_decay = 5e-5              # 1.58e-4 = 3 epoch 0.1 annealing
    l1_regular = 0.0            # weight decay in L1 norm
    l2_regular = 1.e+0          # L2 norm
    label_scalar = 1            # expend from [0, 1]
    flag_add_bn = True
    flag_add_do = False
    np.random.seed(7)           # to repeat results

    if nb_path_for_model == '1':
        model_stacked = build_1path_vgg_bodyonly_model(img_height=img_height,
                                                       weights=initial_weights,
                                                       nb_fc_hidden_layer=nb_hidden_dense_layer,
                                                       nb_fc_hidden_node=nb_hidden_node,
                                                       global_learning_rate=learning_rate,
                                                       learning_rate_multiplier=lr_multiplier,
                                                       weight_decay = w_decay,
                                                       l1_regularization=l1_regular,
                                                       l2_regularization=l2_regular,
                                                       is_bn_enabled=flag_add_bn,
                                                       is_do_enabled=flag_add_do)
        model_path = './models/'
        # # best @ epoch 6
        # weight_filename = 'weights_input448_fc2048bodyonly_imagenet_1125imgx10_ls1_6epoch_sgdlr1e-2m1ae4af01_l1reg0l2reg0_reloc_model.h5'
        # # last @ epoch 8
        # weight_filename = 'weights_input448_fc2048bodyonly_imagenet_1125imgx10_ls1_8epoch_sgdlr1e-2m1ae4af01_l1reg0l2reg0_reloc_model.h5'
        weight_filename = 'weights_input448_2fc2048bodyonly_load_test_model.h5'
        model_stacked.load_weights(model_path + weight_filename)    # by_name = False by default
    elif nb_path_for_model == '2':
        model_stacked = build_2path_vgg_bodytopf_model(img_height=img_height,
                                                       weights=initial_weights,
                                                       nb_fc_hidden_layer=nb_hidden_dense_layer,
                                                       nb_fc_hidden_node=nb_hidden_node,
                                                       global_learning_rate=learning_rate,
                                                       learning_rate_multiplier=lr_multiplier,
                                                       weight_decay = w_decay,
                                                       l1_regularization=l1_regular,
                                                       l2_regularization=l2_regular,
                                                       is_bn_enabled=flag_add_bn,
                                                       is_do_enabled=flag_add_do)
        model_path = './models/'
        weight_filename = 'weights_input448_fc2048bodytopf_imagenet_1125imgx10_ls1_2epoch_sgdlr1e-3m1ae1af01_l1reg0l2reg1_reloc_model.h5'
        model_stacked.load_weights(model_path + weight_filename)
    else:
        raise ValueError("number of stem paths is either '1' or '2'")
    model_stacked.summary()

    img_width = img_height * 5
    batch_size = 1

    if train_valid_split == 'train':
        nb_train_sample = 15182  # take the first part of x10 augmentation
        datagen_train = ImageDataGenerator(rescale=1. / 255)
        generator_train = datagen_train.flow_from_directory(
            'datasets/train_960x1920_20161125/aug_10_times_body_top_concat/',
            target_size=(img_height, img_width),
            batch_size=batch_size,
            shuffle=False,
            class_mode='xy_pos',
            label_file="../../label_list_train1125_15182_x{}.csv"
            .format(label_scalar))
        with Timer('Generate training position xy'):
            train_pos = model_stacked.predict_generator(generator_train,
                                                        val_samples=nb_train_sample)
        # .. 1722 sec for 15182 448 img
        print "train pos xy shape: {}".format(train_pos.shape)
        np.save(open('predicted_data/train_position_result_w20161125img.npy', 'w'), train_pos)
    elif train_valid_split == 'valid':
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
        # .. 225~227 sec for 2000 448 img
        print "valid pos xy shape: {}".format(valid_pos.shape)
        np.save(open('predicted_data/valid_position_result_w20161215img.npy', 'w'), valid_pos)
    else:
        raise ValueError("either go with 'train' or 'valid'")


def visualize(dataset_split):
    if dataset_split == 'train':
        xy = np.load(open('predicted_data/train_position_result_w20161125img.npy'))
        gt = np.loadtxt('datasets/label_list_train1125_15182_x1.csv',
                        dtype='float32', delimiter=',')
    elif dataset_split == 'valid':
        xy = np.load(open('predicted_data/valid_position_result_w20161215img.npy'))
        gt = np.loadtxt('datasets/label_list_valid1215_2000_x1.csv',
                        dtype='float32', delimiter=',')
    else:
        raise ValueError("either go with 'train' or 'valid'")

    # compare
    plt.plot(np.arange(gt.shape[0]) + 1, gt[:, 0], 'b')
    plt.plot(np.arange(gt.shape[0]) + 1, gt[:, 1], 'r')
    plt.plot(np.arange(xy.shape[0]) + 1, xy[:, 0], 'c')
    plt.plot(np.arange(xy.shape[0]) + 1, xy[:, 1], 'orange')
    plt.legend(['x_gt', 'y_gt', 'x_{}'.format(dataset_split), 'y_{}'.format(dataset_split)],
               loc='upper right')
    plt.show()

if __name__ == '__main__':
    flag_dataset_split = 'valid'  # train or valid
    flag_1or2_path_model = '1'
    generate(train_valid_split=flag_dataset_split, nb_path_for_model=flag_1or2_path_model)
    # visualize(flag_dataset_split)
