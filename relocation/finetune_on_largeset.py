__author__ = 'bsl'

from keras.layers import Input
from keras.applications import vgg16
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model
from keras.optimizers import SGD

from load_panoview_label import load_data
import numpy as np
from utils.timer import Timer
from utils.loss_acc_history_rtplot import LossRTPlot


img_width = 224*4
img_height = 224
img_size = (3, img_width, img_height)
input_tensor = Input(batch_shape=(None,) + img_size)

model_vgg_places_notop = vgg16.VGG16(input_tensor=input_tensor, include_top=False)
base_model_output = model_vgg_places_notop.output
base_model_output = Flatten()(base_model_output)
nb_fc_nodes = 1024      # model expension does not affact GPU mem usage
learning_rate_multiplier = 50.0
base_model_output = Dense(nb_fc_nodes,
                          W_learning_rate_multiplier=learning_rate_multiplier,
                          b_learning_rate_multiplier=learning_rate_multiplier,
                          activation='relu')(base_model_output)
base_model_output = Dropout(0.5)(base_model_output)
preds = Dense(2,
              W_learning_rate_multiplier=learning_rate_multiplier,
              b_learning_rate_multiplier=learning_rate_multiplier,
              activation='softmax')(base_model_output)
model_stacked = Model(model_vgg_places_notop.input, preds)
model_path = 'models/'
loaded_model_name = 'train_smallset_vgg2fc1024_places_100epoch_sgdlr1e-5m50_reloc_model.h5'
model_stacked.load_weights(model_path+loaded_model_name)

# no frozen layer

# use 'SGD' with low learning rate
learning_rate = 1e-5
model_stacked.compile(loss='mean_squared_error',
                      optimizer=SGD(lr=learning_rate, momentum=0.9),
                      metrics=[])

with Timer("load pano images"):
    image_array, label_array = load_data()
nb_sample_image = image_array.shape[0]

# train data
batch_size = 32
nb_epoch = 100

loss_rtplot = LossRTPlot()
history_callback = model_stacked.fit(image_array, label_array,
                                     batch_size=batch_size,
                                     nb_epoch=nb_epoch,
                                     validation_split=0.1,  # 10% for validation
                                     validation_data=None,
                                     shuffle=True,
                                     callbacks=[loss_rtplot])

# record the loss
record = np.column_stack((np.array(history_callback.epoch) + 1,
                          history_callback.history['loss'],
                          history_callback.history['val_loss']))

np.savetxt('training_procedure/convergence_largeset_vgg2fc{}_places_{}epoch_sgdlr{}m{}_reloc_model.csv'
           .format(nb_fc_nodes, (history_callback.epoch[-1]+1), learning_rate, learning_rate_multiplier),
           record, delimiter=',')
model_stacked.save_weights('models/train_largeset_vgg2fc{}_places_{}epoch_sgdlr{}m{}_reloc_model.h5'
                           .format(nb_fc_nodes,
                                   (history_callback.epoch[-1]+1),
                                   learning_rate,
                                   learning_rate_multiplier))