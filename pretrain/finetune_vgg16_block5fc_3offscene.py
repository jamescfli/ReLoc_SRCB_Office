__author__ = 'bsl'

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
# from vgg_conv_layers import vgg_conv_layers
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16
from keras.layers import Input
from keras.models import Model
# from keras import backend as K
import h5py

# # debug
# import os
# os.getcwd()
# os.chdir('/home/bsl/JmsLi/PythonProjects/ReLoc_SRCB_Office/pretrain')

img_width = 150
img_height = 150
# model_vgg = vgg_conv_layers(img_size=(img_width, img_height))
img_size = (3, img_width, img_height)
input_tensor = Input(batch_shape=(None,) + img_size)
# input_tensor = K.placeholder(shape=(None, 3, img_width, img_height)) # this does not work
model_vgg = vgg16.VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)

# top_model = Sequential()
# top_model.add(Flatten(input_shape=model_vgg.output_shape[1:]))
# top_model.add(Dense(256, activation='relu'))    # other: 512, 1024, 4096 (Imagenet 1000 cls)
# top_model.add(Dropout(0.5))
# top_model.add(Dense(3, activation='softmax'))   # conf, off, off_cub
# top_model.load_weights('models/bottleneck_fc_classifier_model.h5')
# model_vgg.add(top_model)

base_model_output = model_vgg.output
base_model_output = Flatten()(base_model_output)
base_model_output = Dense(256, activation='relu')(base_model_output)
base_model_output = Dropout(0.5)(base_model_output)
preds = Dense(3, activation='softmax')(base_model_output)

model_stacked = Model(model_vgg.input, preds)

f = h5py.File("models/bottleneck_fc_classifier_model.h5", "r")
# .. f.attrs['layer_names'] = array(['flatten_1', 'dense_1', 'dropout_1', 'dense_2'], dtype='|S9')
# .. only dense_1 and dense_2 need to be loaded
# flatten_1 layer : model_stacked.layers[19].get_weights() = []
# dense_1 layer : dense_1_W model_stacked.layers[20].get_weights()[0].shape = (8192, 256) and
#                 dense_1_b model_stacked.layers[20].get_weights()[1].shape = (256,)
# dropout_1 layer : model_stacked.layers[21].get_weights() = []
# dense_2 layer : dense_2_W model_stacked.layers[22].get_weights()[0].shape = (256, 3) and
#                 dense_2_b model_stacked.layers[22].get_weights()[1].shape = (3,)
old_w = model_stacked.layers[20].get_weights()[0]
old_b = model_stacked.layers[20].get_weights()[1]
g = f['dense_1']
weights_dense_1 = [g['dense_1_W'], g['dense_1_b']]
model_stacked.layers[20].set_weights(weights_dense_1)
# new_w = model_stacked.layers[20].get_weights()[0]
# new_b = model_stacked.layers[20].get_weights()[1]
g = f['dense_2']
weights_dense_2 = [g['dense_2_W'], g['dense_2_b']]
model_stacked.layers[22].set_weights(weights_dense_2)

# for saved_layer_index in range(4):   # 19-flatten_1, 20-dense_1, 21-dropout_1, 22-dense_2
#     # 0~3 --> 19~22 totally 23 layer
#     stacked_layer_index = saved_layer_index+19
#     weights = top_model.layers[saved_layer_index].get_weights()
#     model_stacked.layers[stacked_layer_index].set_weights(weights)
print('Top FC layers loaded with pretrained values in .h5 file')

# # freeze the first 4 convolution blocks, up to 25 layers
# for layer in model_stacked.layers[:25]:   # for 'from vgg_conv_layers import vgg_conv_layers'
#     layer.trainable = False
# .. model is ready to go
# .. ?? why 25, all layer freezed get low accuracy. Short answer: no zeropadding anymore
# layer counting is quite different from previous ones where ZeroPadding2D was counted as layers
# in "from keras.applications import vgg16" only conv and maxpooling layers are counted, totally 18 layers before fc layer

# model_stacked.layers.__len__() = 23 layers = 19 + Flatten + Dense(256) + Dropout(0.5) + Dense(3)
# model_vgg.layers.__len__() = 19 layers

# reset trainable layers in VGG16 from keras.applications
for layer in model_stacked.layers[:15]:     # for 'from keras.applications import vgg16', topped by 23 layers in this case
    layer.trainable = False
# 15 for training block5 and fc layers
# 11 for training block5 block4 and fc layers

# discard 'rmsprop' and use 'SGD' with very low learning rate to avoid overfitting
model_stacked.compile(loss='categorical_crossentropy',      # for classification
                      optimizer=SGD(lr=1e-4, momentum=0.9),
                      metrics=['accuracy'])

# prepare data
# train
datagen_train = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
generator_train = datagen_train.flow_from_directory('data_large_3classes/train',
                                                    target_size=(img_height,img_width),
                                                    batch_size=32,
                                                    class_mode='categorical')
# test
datagen_test = ImageDataGenerator(rescale=1./255)
generator_test = datagen_test.flow_from_directory('data_large_3classes/validation',
                                                  target_size=(img_height,img_width),
                                                  batch_size=32,
                                                  class_mode='categorical')
nb_epoch = 50
model_stacked.fit_generator(generator_train,
                        samples_per_epoch=4000,     # 4000 for training, 1000 for validation
                        nb_epoch=nb_epoch,          # default 50
                        validation_data=generator_test,
                        nb_val_samples=1000)

# check whether new_w and new_b have been changed in layer[20], check layer.trainable
# new_w = model_stacked.layers[20].get_weights()[0]
# new_b = model_stacked.layers[20].get_weights()[1]
# new_w - old_w = 0 and new_b - old_b = 0
# layer.trainable = False
# The variation on training and validation set is simply because of the random Data Augmentation in generator

# save the pretrained parameter into models folder
# model_stacked.save_weights('models/vgg_block5fc_finetuned_classifier_model.h5')