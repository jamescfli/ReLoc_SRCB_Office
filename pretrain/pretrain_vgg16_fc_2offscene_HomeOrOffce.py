__author__ = 'bsl'

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator, img_to_array
import numpy as np
from keras.applications import vgg16
from keras.layers import Input
import time

img_width = 224
img_height = 224
img_size = (3, img_width, img_height)
input_tensor = Input(batch_shape=(None,) + img_size)
model_vgg = vgg16.VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)

start_time = time.time()    # timer

batch_size = 64    # used to be 32
datagen_train = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
generator_train = datagen_train.flow_from_directory('datasets/data_256_HomeOrOff/train',
                                              target_size=(img_height,img_width),
                                              batch_size=batch_size,
                                              class_mode=None,  # no labels, just data
                                              shuffle=False)
# total number of training image are 21244+31955=53199 and 100+100=200 for testing
bottleneck_features_train = model_vgg.predict_generator(generator_train, 21244+31955)
np.save(open('feature_map/bottleneck_features/train_2classes_HomeOrOff_fmap.npy', 'w'), bottleneck_features_train)

datagen_test = ImageDataGenerator(rescale=1./255)
generator_test = datagen_test.flow_from_directory('datasets/data_256_HomeOrOff/test',
                                                  target_size=(img_height,img_width),
                                                  batch_size=batch_size,
                                                  class_mode=None,
                                                  shuffle=False)
bottleneck_features_test = model_vgg.predict_generator(generator_test, 200)
np.save(open('feature_map/bottleneck_features/test_2classes_HomeOrOff_fmap.npy', 'w'), bottleneck_features_test)

# train a fully connected network, Flatten+Dense+Dropout+Dense(SoftMax) with fmap
data_train = np.load(open('feature_map/bottleneck_features/train_2classes_HomeOrOff_fmap.npy'))
labels_train = np.array([1,0] * 21244 + [0,1] * 31955).reshape(21244+31955, 2)

data_test = np.load(open('feature_map/bottleneck_features/test_2classes_HomeOrOff_fmap.npy'))
labels_test = np.array([1,0] * 100 + [0,1] * 100).reshape(200, 2)

end_time = time.time()
print "time lapse when preparing data: {} sec".format(end_time - start_time)
# e.g. 486 sec - ~8 min

model_fc = Sequential()
model_fc.add(Flatten(input_shape=data_train.shape[1:]))
model_fc.add(Dense(256, activation='relu'))
model_fc.add(Dropout(0.5))
model_fc.add(Dense(2, activation='softmax'))

model_fc.compile(optimizer='rmsprop',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
nb_epoch = 20   # just for pretrain, shadow network can easily converge within small number of epochs
model_fc.fit(data_train, labels_train,
             nb_epoch=nb_epoch, batch_size=batch_size,
             validation_data=(data_test, labels_test))
# model_fc.save_weights('models/bottleneck_fc_2class_HomeOrOff_{}epochs_model.h5'.format(nb_epoch))
model_fc.save_weights('models/bottleneck_fc_2class_HomeOrOff_model.h5')  # requried by fine-tuning