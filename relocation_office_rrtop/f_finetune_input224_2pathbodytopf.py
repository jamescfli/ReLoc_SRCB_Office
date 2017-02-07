__author__ = 'bsl'

from relocation_office_rrtop.d_build_parallel_model_bodytop import build_2path_vgg_bodytopf_model

from utils.custom_image import ImageDataGenerator
# from utils.loss_acc_mse_history_rtplot import LossMseRTPlot
from utils.lr_annealing import LearningRateAnnealing
from keras.callbacks import ModelCheckpoint

import numpy as np
import os


if __name__ == '__main__':
    img_height = 224
    initial_weights = 'imagenet'
    nb_hidden_node = 2048       # where fc layer for topf will be divided by 4, i.e. 512
    learning_rate = 1.e-2       # to conv layers
    lr_multiplier = 1.0         # to top fc layers
    l1_regular = 0.0            # weight decay in L1 norm
    l2_regular = 1.e+0          # L2 norm
    label_scalar = 10           # expend from [0, 1]
    flag_add_bn = True
    flag_add_do = True
    do_ratio = 0.5
    batch_size = 32             # tried 32 (224), 3850MB
    nb_epoch = 12               # due to higher dimension of 448 img @ network bottle-neck
    nb_epoch_annealing = 3     # anneal for every <> epochs
    annealing_factor = 0.1
    np.random.seed(7)           # to repeat results
    model_stacked = build_2path_vgg_bodytopf_model(img_height=img_height,
                                                   weights=initial_weights,
                                                   nb_fc_hidden_node=nb_hidden_node,
                                                   dropout_ratio=do_ratio,
                                                   global_learning_rate=learning_rate,
                                                   learning_rate_multiplier=lr_multiplier,
                                                   l1_regularization=l1_regular,
                                                   l2_regularization=l2_regular,
                                                   is_bn_enabled=flag_add_bn,
                                                   is_do_enabled=flag_add_do)
    model_stacked.summary()

    nb_train_sample = 15182
    aug_factor = 10     # 1ep ~= 10ep before
    img_width = 5*img_height

    datagen_train = ImageDataGenerator(rescale=1. / 255)
    generator_train = datagen_train.flow_from_directory(
        'relocation_office_rrtop/datasets/train_960x1920_20161125/aug_10_times_body_top_concat/',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=True,
        class_mode='xy_pos',
        label_file="../../label_list_train1125_15182_aug{}_x{}.csv".format(aug_factor, label_scalar))

    nb_valid_sample = 2000
    datagen_valid = ImageDataGenerator(rescale=1. / 255)
    generator_test = datagen_valid.flow_from_directory(
        'relocation_office_rrtop/datasets/valid_480x2400_concat_nb2000_20161215/concat/',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=False,
        class_mode='xy_pos',
        label_file="../../label_list_valid1215_2000_x{}.csv".format(label_scalar))

    # model training callbacks
    # # 1) plot mse graphs
    # loss_mse_rtplot = LossMseRTPlot()
    # 2) lr annealing
    annealing_schedule = LearningRateAnnealing(nb_epoch_annealing, annealing_factor)
    # 3) checkpoint saving in case of outage
    saver_filepath = 'relocation_office_rrtop/model_checkpoints'
    saver_filename = 'model_latest_{epoch:03d}.h5'  # currently < 1000 epochs
    checkpoint_saver = ModelCheckpoint(saver_filepath + os.path.sep + saver_filename,
                                       monitor='val_mean_squared_error',
                                       verbose=1,
                                       save_best_only=False,
                                       save_weights_only=True)

    history_callback = model_stacked.fit_generator(generator_train,
                                                   samples_per_epoch=nb_train_sample*aug_factor,
                                                   nb_epoch=nb_epoch,
                                                   validation_data=generator_test,
                                                   nb_val_samples=nb_valid_sample,
                                                   callbacks=[annealing_schedule, checkpoint_saver],
                                                   verbose=1)

    # record
    record = np.column_stack((np.array(history_callback.epoch) + 1,
                              history_callback.history['loss'],
                              history_callback.history['val_loss'],
                              history_callback.history['mean_squared_error'],
                              history_callback.history['val_mean_squared_error']))

    np.savetxt(
        'relocation_office_rrtop/training_procedure/convergence_input{}_fc{}body_div4topf_{}_1125imgx{}_ls{}_{}epoch_sgdlr{:.0e}m{}ae{}af{}_l1reg{:.0e}l2reg{:.0e}_reloc_model.csv'
            .format(img_height,
                    nb_hidden_node,
                    initial_weights,
                    aug_factor,
                    label_scalar,
                    nb_epoch,
                    learning_rate,
                    int(lr_multiplier),
                    nb_epoch_annealing,  # aepoch
                    annealing_factor,  # afactor
                    l1_regular,
                    l2_regular),
        record, delimiter=',')

    model_stacked.save_weights(
        'relocation_office_rrtop/models/weights_input{}_fc{}body_div4topf_{}_1125imgx{}_ls{}_{}epoch_sgdlr{:.0e}m{}ae{}af{}_l1reg{:.0e}l2reg{:.0e}_reloc_model.h5'
            .format(img_height,
                    nb_hidden_node,
                    initial_weights,
                    aug_factor,
                    label_scalar,
                    nb_epoch,
                    learning_rate,
                    int(lr_multiplier),
                    nb_epoch_annealing,  # aepoch
                    annealing_factor,  # afactor
                    l1_regular,
                    l2_regular), overwrite=False)
