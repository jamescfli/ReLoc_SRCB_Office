__author__ = 'bsl'

from keras.layers.core import Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16
from keras.layers import Input
from keras.models import Model
import h5py

# # debug
# import os
# os.getcwd()
# os.chdir('/home/bsl/JmsLi/PythonProjects/ReLoc_SRCB_Office/pretrain')

# original size for train challenge is 256x256
img_width = 224
img_height = 224
img_size = (3, img_width, img_height)
input_tensor = Input(batch_shape=(None,) + img_size)
model_vgg = vgg16.VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)

base_model_output = model_vgg.output
base_model_output = Flatten()(base_model_output)
base_model_output = Dense(256, activation='relu')(base_model_output)
base_model_output = Dropout(0.5)(base_model_output)
preds = Dense(2, activation='softmax')(base_model_output)
# .. 2 scenes considered: home_office and office

model_stacked = Model(model_vgg.input, preds)

f = h5py.File("models/bottleneck_fc_2class_HomeOrOff_model.h5", "r")
g = f['dense_1']
weights_dense_1 = [g['dense_1_W'], g['dense_1_b']]
model_stacked.layers[20].set_weights(weights_dense_1)
g = f['dense_2']
weights_dense_2 = [g['dense_2_W'], g['dense_2_b']]
model_stacked.layers[22].set_weights(weights_dense_2)
print('Top FC layers loaded with pretrained values in .h5 file')

# reset trainable layers in VGG16 from keras.applications
for layer in model_stacked.layers[:15]:
    layer.trainable = False
# 15 - train block5 and fc layers
# 11 - train block5,4 and fc layers

# use 'SGD' with low learning rate
model_stacked.compile(loss='categorical_crossentropy',
                      optimizer=SGD(lr=1e-4, momentum=0.9),
                      metrics=['accuracy'])

# train data
batch_size = 64    # used to be 32
datagen_train = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
generator_train = datagen_train.flow_from_directory('datasets/data_256_HomeOrOff/train',
                                                    target_size=(img_height,img_width),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')
# test
datagen_test = ImageDataGenerator(rescale=1./255)
generator_test = datagen_test.flow_from_directory('datasets/data_256_HomeOr/test',
                                                  target_size=(img_height,img_width),
                                                  batch_size=batch_size,
                                                  class_mode='categorical')
nb_epoch = 1                # e.g. 50
nb_train_samples = 53199    # 21244+31955=53199
nb_test_samples = 200       # 100x2
model_stacked.fit_generator(generator_train,
                            samples_per_epoch=nb_train_samples,  # normally equal to nb of training samples
                            nb_epoch=nb_epoch,
                            validation_data=generator_test,
                            nb_val_samples=nb_test_samples)

# save the pretrained parameter into models folder
model_stacked.save_weights('models/vgg_block5fc_finetuned_4class_model.h5')