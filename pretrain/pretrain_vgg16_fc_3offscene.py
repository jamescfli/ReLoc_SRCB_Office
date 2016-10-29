__author__ = 'bsl'
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator, img_to_array
import numpy as np
# from vgg_conv_layers import vgg_conv_layers
from keras.applications import vgg16
from keras.layers import Input

img_width = 150
img_height = 150
img_size = (3, img_width, img_height)
input_tensor = Input(batch_shape=(None,) + img_size)
model_vgg = vgg16.VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)
# model_vgg = vgg_conv_layers(img_size=(img_width, img_height))
# compile is for configuring the model for training, we haven't decided how to train the whole network

datagen_train = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

generator_train = datagen_train.flow_from_directory('datasets/data_large_3classes/train',
                                              target_size=(img_height,img_width),
                                              batch_size=32,
                                              class_mode=None,  # no labels, just data
                                              shuffle=False)
# calculate the feature map for training data
bottleneck_features_train = model_vgg.predict_generator(generator_train, 12000)
# .. (12000, 512, 4, 4) - 4000 * (conf, off, off_cub), 512 kernels, 4*4 feature map size
# save output as a NumPy array
np.save(open('feature_map/bottleneck_features/bottleneck_features_train.npy', 'w'), bottleneck_features_train)

datagen_test = ImageDataGenerator(rescale=1./255)
generator_test = datagen_test.flow_from_directory('datasets/data_large_3classes/validation',
                                                  target_size=(img_height,img_width),
                                                  batch_size=32,
                                                  class_mode=None,
                                                  shuffle=False)
# calculate the feature map for testing data
bottleneck_features_test = model_vgg.predict_generator(generator_test, 3000)    # leftover batch size 24
np.save(open('feature_map/bottleneck_features/bottleneck_features_test.npy', 'w'), bottleneck_features_test)
# .. (800, 512, 4, 4)

# at this point, we just apply the net to generate feature maps

# train a totally new model with fully connected layer
data_train = np.load(open('feature_map/bottleneck_features/bottleneck_features_train.npy'))
labels_train = np.array([1,0,0] * 4000 + [0,1,0] * 4000 + [0,0,1] * 4000).reshape(12000, 3)

data_test = np.load(open('feature_map/bottleneck_features/bottleneck_features_test.npy'))
labels_test = np.array([1,0,0] * 1000 + [0,1,0] * 1000 + [0,0,1] * 1000).reshape(3000, 3)

model_fc = Sequential()
model_fc.add(Flatten(input_shape=data_train.shape[1:])) # shape cut sample index, get the shape of feature map
model_fc.add(Dense(256, activation='relu'))
model_fc.add(Dropout(0.5))
model_fc.add(Dense(3, activation='softmax'))

model_fc.compile(optimizer='rmsprop',   # faster than SGD
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
model_fc.fit(data_train, labels_train,
             nb_epoch=50, batch_size=32,
             validation_data=(data_test, labels_test))
model_fc.save_weights('models/bottleneck_fc_classifier_model.h5')  # requried by fine-tuning