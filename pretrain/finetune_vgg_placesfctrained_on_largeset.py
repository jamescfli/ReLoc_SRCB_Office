__author__ = 'bsl'

from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout
from keras.constraints import maxnorm
from keras.applications import vgg16
from keras.optimizers import SGD
from keras.models import model_from_json

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
                      global_learning_rate=1e-5,
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
    vgg_model_output = Dense(nb_fc_hidden_node,
                               name='FC_Dense_2',
                               W_constraint=maxnorm(weight_constraint),
                               W_learning_rate_multiplier=learning_rate_multiplier,
                               b_learning_rate_multiplier=learning_rate_multiplier,
                               activation='relu')(vgg_model_output)
    vgg_model_output = Dropout(dropout_ratio)(vgg_model_output)
    vgg_model_output = Dense(2,
                               name='FC_Dense_3',
                               W_learning_rate_multiplier=learning_rate_multiplier,
                               b_learning_rate_multiplier=learning_rate_multiplier,
                               activation='linear')(vgg_model_output)
    vgg_model_withtop = Model(vgg_places_model_notop.input, vgg_model_output)
    vgg_model_withtop.load_weights('models/train_input224_topfc1024_smallset_100epoch_DO0.5_WC2_HomeOrOff_model.h5',
                             by_name=True)

    # set frozen layers
    for layer in vgg_model_withtop.layers[:nb_frozen_layer]:
        layer.trainable = False

    vgg_model_withtop.compile(loss='categorical_crossentropy',
                              optimizer=SGD(lr=global_learning_rate, momentum=0.9),
                              # optimizer='rmsprop',
                              metrics=['accuracy'])
    return vgg_model_withtop      # total 26 layers


def load_vggfc_model(model_structure_path=None,
                     model_weight_path=None,
                     global_learning_rate=1e-05):
    # load structure
    json_file = open(model_structure_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    vgg_model_withtop = model_from_json(loaded_model_json)
    # load weights
    vgg_model_withtop.load_weights("model_weight_path")
    print "load " + model_structure_path + " and " + model_weight_path + " from disk"
    vgg_model_withtop.compile(loss='categorical_crossentropy',
                              optimizer=SGD(lr=global_learning_rate, momentum=0.9),
                              # optimizer='adadelta',
                              metrics=['accuracy'])
    print 'model compiled'
    return vgg_model_withtop


# build model from scratch
nb_hidden_node = 1024
do_ratio = 0.5
weight_con = 2
nb_fzlayer = 11         # 11 block4, 15 block5, 19 top fc
learning_rate = 1e-4    # to conv layers
lr_multiplier = 10.0     # to top fc layers
model_stacked = build_vggfc_model(nb_fc_hidden_node=nb_hidden_node,
                                  dropout_ratio=do_ratio,
                                  weight_constraint=weight_con,
                                  nb_frozen_layer=nb_fzlayer,
                                  global_learning_rate=learning_rate,
                                  learning_rate_multiplier=lr_multiplier)
# # build model from trained one
# model_struct_path = "models/structure..json"
# model_wt_path = 'models/weights..h5'
# model_stacked = load_vggfc_model(model_structure_path=model_struct_path,
#                                    model_weight_path=model_wt_path,
#                                    global_learning_rate=learning_rate)


batch_size = 32
nb_epoch = 100      # 537s/epoch, 15 hours * 60 *60 /537 = 100.56 epochs

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
                                                  shuffle=False,
                                                  class_mode='categorical')

# fit model
loss_acc_rtplot = LossAccRTPlot()
history_callback = model_stacked.fit_generator(generator_train,
                                               samples_per_epoch=nb_train_sample,
                                               nb_epoch=nb_epoch,
                                               validation_data=generator_test,
                                               nb_val_samples=nb_test_sample,
                                               # callbacks=[])
                                               callbacks=[loss_acc_rtplot])

# record
record = np.column_stack((np.array(history_callback.epoch) + 1,
                          history_callback.history['loss'],
                          history_callback.history['val_loss'],
                          history_callback.history['acc'],
                          history_callback.history['val_acc']))

np.savetxt('training_procedure/convergence_vgg3fc{}_largeset_{}fzlayer_{}epoch_sgdlr{}m{}_HomeOrOff_model.csv'
           .format(nb_hidden_node,
                   nb_fzlayer,
                   (history_callback.epoch[-1]+1),
                   learning_rate,
                   int(lr_multiplier)),
           record, delimiter=',')
model_stacked_json = model_stacked.to_json()
with open('models/structure_vgg3fc{}_largeset_{}fzlayer_{}epoch_sgdlr{}m{}_HomeOrOff_model.h5'
                  .format(nb_hidden_node,
                          nb_fzlayer,
                          (history_callback.epoch[-1]+1),
                          learning_rate,
                          lr_multiplier), "w") \
        as json_file_model_stacked:
    json_file_model_stacked.write(model_stacked_json)
model_stacked.save_weights('models/weights_vgg3fc{}_largeset_{}fzlayer_{}epoch_sgdlr{}m{}_HomeOrOff_model.h5'
                           .format(nb_hidden_node,
                                   nb_fzlayer,
                                   (history_callback.epoch[-1]+1),
                                   learning_rate,
                                   lr_multiplier))
