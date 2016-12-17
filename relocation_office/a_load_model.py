__author__ = 'bsl'

from keras.layers import Input
from keras.applications import vgg16

import numpy as np


img_height = 448  # options: 448, 224, and original 450*1920
img_width = img_height * 4

img_size = (3, img_height, img_width)
input_tensor = Input(batch_shape=(None,) + img_size)
vgg_office_model_notop = vgg16.VGG16(input_tensor=input_tensor, include_top=False)
# debug: verify loading weights
w_init_weights = vgg_office_model_notop.layers[17].get_weights()[0]     # (512, 512, 3, 3)
print 'w shape: {}'.format(w_init_weights.shape)
b_init_weights = vgg_office_model_notop.layers[17].get_weights()[1]     # (512,)

model_path = 'models/'
model_weight_filename = 'weights_vgg2fc256_largeset_11fzlayer_60epoch_sgdlr5e-5m10anneal20epoch_HomeOrOff_model.h5'
vgg_office_model_notop.load_weights(model_path+model_weight_filename, by_name=True)
# debug: verify loading weights
w_load_weights = vgg_office_model_notop.layers[17].get_weights()[0]
b_load_weights = vgg_office_model_notop.layers[17].get_weights()[1]
print w_init_weights.shape
print w_init_weights.shape == w_load_weights.shape      # True
print np.array_equal(w_init_weights, w_load_weights)    # False
print b_init_weights.shape == b_load_weights.shape      # True
print np.array_equal(b_init_weights, b_load_weights)    # False

# verify
from utils.model_converter.compare_model_parameters import equal_model
# print equal_model(vgg16.VGG16(input_tensor=input_tensor, include_top=False),
#                   vgg16.VGG16(input_tensor=input_tensor, include_top=False))
print equal_model(vgg_office_model_notop,
                  vgg16.VGG16(input_tensor=input_tensor, include_top=False))    # False
vgg_office_model_notop.summary()