__author__ = 'bsl'

from keras.layers import Input
from keras.applications import vgg16
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model
from keras.optimizers import SGD

from load_panoview_label import load_data
import numpy as np
from utils.timer import Timer


img_width = 224*4
img_height = 224
img_size = (3, img_width, img_height)
input_tensor = Input(batch_shape=(None,) + img_size)

model_vgg_places_notop = vgg16.VGG16(input_tensor=input_tensor, include_top=False)
model_vgg_places_notop.load_weights('models/vgg16_places365_notop.h5')
# load vgg net without top
base_model_output = model_vgg_places_notop.output
base_model_output = Flatten()(base_model_output)
nb_fc_nodes = 1024      # model expension does not affact GPU mem usage
base_model_output = Dense(nb_fc_nodes,
                          W_learning_rate_multiplier=50.0,
                          b_learning_rate_multiplier=50.0,
                          activation='relu')(base_model_output)
base_model_output = Dropout(0.5)(base_model_output)
preds = Dense(2,
              W_learning_rate_multiplier=50.0,
              b_learning_rate_multiplier=50.0,
              activation='softmax')(base_model_output)
model_stacked = Model(model_vgg_places_notop.input, preds)

# no frozen layer

# use 'SGD' with low learning rate
learning_rate = 1e-5
model_stacked.compile(loss='mean_squared_error',
                      optimizer=SGD(lr=learning_rate, momentum=0.9),
                      metrics=[])   # loss = Euclidean distance
with Timer("load pano images"):
    image_array, label_array = load_data()
nb_sample_image = image_array.shape[0]
nb_sample_smallset = 2000
rand_sample_index = np.random.choice(nb_sample_image, nb_sample_smallset, replace = False)
image_array = image_array[rand_sample_index, :, :, :]
label_array = label_array[rand_sample_index, :]


# train data
batch_size = 32
nb_epoch = 100

history_callback = model_stacked.fit(image_array, label_array,
                                     batch_size=batch_size,
                                     nb_epoch=nb_epoch,
                                     validation_split=0.0,
                                     validation_data=None,
                                     shuffle=True)
