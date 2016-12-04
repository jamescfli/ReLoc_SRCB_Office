__author__ = 'bsl'

from keras.applications import vgg16
from keras.layers import Input


img_width = 224
img_height = 224
img_size = (3, img_width, img_height)
input_tensor = Input(batch_shape=(None,) + img_size)
vgg_places_model_notop = vgg16.VGG16(input_tensor=input_tensor, include_top=False)
vgg_places_model_notop.load_weights('models/vgg16_places365_notop_newpool.h5')

# run the following lines in latest Keras, not MarcBS
vgg_places_model_notop.save_weights('models/vgg16_places365_notop_nowarning.h5')