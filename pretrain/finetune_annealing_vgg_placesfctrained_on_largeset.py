__author__ = 'bsl'

from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout
from keras.constraints import maxnorm
from keras.applications import vgg16
from keras.optimizers import SGD

from utils.custom_image import ImageDataGenerator
from utils.loss_acc_history_rtplot import LossAccRTPlot

import numpy as np


img_height = 224
img_width = 224


def build_vggfc_model(vgg_initial_weights='places',
                      nb_fc_hidden_node=1024,
                      dropout_ratio=0.5,
                      weight_constraint=2,
                      nb_frozen_layer=0,
                      learning_rate_multiplier=1.0):
    img_size = (3, img_height, img_width)  # expected: shape (nb_sample, 3, 480, 1920)
    input_tensor = Input(batch_shape=(None,) + img_size)

    vgg_places_model_notop = vgg16.VGG16(input_tensor=input_tensor, include_top=False)
    if vgg_initial_weights == 'places':
        print 'loading places weights ..'
        vgg_places_model_notop.load_weights('models/vgg16_places365_notop_weights.h5')
    else:   # o.w leave it as ImageNet
        print 'keep using imagenet weights ..'
    vgg_model_output = vgg_places_model_notop.output
    vgg_model_output = Flatten()(vgg_model_output)

    vgg_model_output = Dense(nb_fc_hidden_node,
                             name='FC_Dense_1',
                             W_constraint=maxnorm(weight_constraint),
                             W_learning_rate_multiplier=learning_rate_multiplier,
                             b_learning_rate_multiplier=learning_rate_multiplier,
                             activation='relu')(vgg_model_output)
    vgg_model_output = Dropout(dropout_ratio)(vgg_model_output)
    vgg_model_output = Dense(2,
                             name='FC_Dense_2',
                             W_learning_rate_multiplier=learning_rate_multiplier,
                             b_learning_rate_multiplier=learning_rate_multiplier,
                             activation='softmax')(vgg_model_output)
    vgg_model_withtop = Model(vgg_places_model_notop.input, vgg_model_output)
    vgg_model_withtop.load_weights('models/train_input224_top2fc256_largeset_100epoch_DO0.5_WC2_sgd1e-5_HomeOrOff_model.h5',
                                   by_name=True)

    # set frozen layers
    for layer in vgg_model_withtop.layers[:nb_frozen_layer]:
        layer.trainable = False

    # leave compile outside

    return vgg_model_withtop      # total 26 layers


# build model from scratch
nb_hidden_node = 256
do_ratio = 0.5
weight_con = 2
nb_fzlayer = 15         # 11 block4, 15 block5, 19 top fc
lr_multiplier = 1.0     # to top fc layers
# initial training
model_stacked = build_vggfc_model(nb_fc_hidden_node=nb_hidden_node,
                                  dropout_ratio=do_ratio,
                                  weight_constraint=weight_con,
                                  nb_frozen_layer=nb_fzlayer,
                                  learning_rate_multiplier=lr_multiplier)
print model_stacked.summary()

batch_size = 32
nb_epoch = 60          # 356s/epoch
learning_rate = 5e-5    # initial learning rate

# prepare training data
nb_train_sample = 18344+29055

datagen_train = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
generator_train = datagen_train.flow_from_directory('datasets/data_256_HomeOrOff/train/',
                                                    target_size=(img_height, img_width),
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    class_mode='categorical')

# prepare test data
nb_test_sample = 3000*2
datagen_test = ImageDataGenerator(rescale=1./255)   # keep consistent validation
generator_test = datagen_test.flow_from_directory('datasets/data_256_HomeOrOff/test/',
                                                  target_size=(img_height, img_width),
                                                  batch_size=batch_size,
                                                  shuffle=False,    # one loss/acc value, no need for shuffle
                                                  class_mode='categorical')

# fine tune with annealing
nb_epoch_per_stage = 20
nb_stage = nb_epoch/nb_epoch_per_stage
record = np.zeros((nb_epoch, 5), dtype='float32')
for stage in np.arange(nb_stage):
    learning_rate_stage = learning_rate / (2.0**stage)  # halved learning rate for every 'nb_epoch_per_stage' epochs
    print "learning rate for this stage: {}".format(learning_rate_stage)
    model_stacked.compile(loss='categorical_crossentropy',
                          optimizer=SGD(lr=learning_rate_stage, momentum=0.9),
                          metrics=['accuracy'])
    history_callback = model_stacked.fit_generator(generator_train,
                                                   samples_per_epoch=nb_train_sample,
                                                   nb_epoch=nb_epoch_per_stage,
                                                   validation_data=generator_test,
                                                   nb_val_samples=nb_test_sample)
    # record
    record_per_stage = np.column_stack((np.array(history_callback.epoch) + 1 + stage*nb_epoch_per_stage,
                                        history_callback.history['loss'],
                                        history_callback.history['val_loss'],
                                        history_callback.history['acc'],
                                        history_callback.history['val_acc']))
    record[(stage*nb_epoch_per_stage):((stage+1)*nb_epoch_per_stage), :] = record_per_stage


np.savetxt('training_procedure/convergence_vgg2fc{}_largeset_{}fzlayer_{}epoch_sgdlr{}m{}anneal{}epoch_HomeOrOff_model.csv'
           .format(nb_hidden_node,
                   nb_fzlayer,
                   nb_epoch,
                   learning_rate,
                   lr_multiplier,
                   nb_epoch_per_stage),
           record, delimiter=',')
model_stacked_json = model_stacked.to_json()
# with open('models/structure_vgg2fc{}_largeset_{}fzlayer_{}epoch_sgdlr{}m{}anneal{}epoch_HomeOrOff_model.json'
#                   .format(nb_hidden_node,
#                           nb_fzlayer,
#                           nb_epoch,
#                           learning_rate,
#                           lr_multiplier,
#                           nb_epoch_per_stage), "w") \
#         as json_file_model_stacked:
#     json_file_model_stacked.write(model_stacked_json)
model_stacked.save_weights('models/weights_vgg2fc{}_largeset_{}fzlayer_{}epoch_sgdlr{}m{}anneal{}epoch_HomeOrOff_model.h5'
                           .format(nb_hidden_node,
                                   nb_fzlayer,
                                   nb_epoch,
                                   learning_rate,
                                   lr_multiplier,
                                   nb_epoch_per_stage))
model_stacked.save('models/fullinfo_vgg2fc{}_largeset_{}fzlayer_{}epoch_sgdlr{}m{}anneal{}epoch_HomeOrOff_model.h5'
                   .format(nb_hidden_node,
                           nb_fzlayer,
                           nb_epoch,
                           learning_rate,
                           lr_multiplier,
                           nb_epoch_per_stage))
