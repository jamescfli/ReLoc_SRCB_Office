__author__ = 'bsl'

from keras.applications import vgg16
from keras.layers import Input
from keras.models import model_from_json


img_width = 224
img_height = 224
img_size = (3, img_width, img_height)
input_tensor = Input(batch_shape=(None,) + img_size)

# model 1
print 'load vgg with imagenet parameters'
vgg_imagenet_model_notop = vgg16.VGG16(input_tensor=input_tensor, include_top=False)

# model 2
print 'load vgg with places365 parameters'
vgg_places_model_notop = vgg16.VGG16(input_tensor=input_tensor, include_top=False)
vgg_places_model_notop.load_weights('models/vgg16_places365_notop_weights.h5')

# model 3
print 'load vgg from json with places parameters'
json_file = open('models/vgg16_places365_notop_structure.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
vgg_places_model_notop_fromJson = model_from_json(loaded_model_json)
vgg_places_model_notop_fromJson.load_weights('models/vgg16_places365_notop_weights.h5')

# compare models
from compare_model_parameters import equal_model
print equal_model(vgg_imagenet_model_notop, vgg_places_model_notop)         # False
print equal_model(vgg_places_model_notop, vgg_places_model_notop_fromJson)  # True
print vgg_places_model_notop_fromJson.summary()
