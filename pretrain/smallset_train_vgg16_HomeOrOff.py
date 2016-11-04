from __future__ import print_function
from keras.utils import np_utils
from keras.layers import Input
from keras.applications import vgg16
from keras.layers.core import Flatten, Dense, Dropout
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras import backend as K
import os

# differentiate home-office and office in a very small data set, 1000*2

batch_size = 32
nb_classes = 2
nb_img_per_class = 1000
nb_epoch = 200      # frz 11 layers, 63sec/epoch, *810 = ~14hours
data_augmentation = True

img_width, img_height = 224, 224
img_channels = 3    # RGB

def load_data(path=None, grayscale=False, target_size=(img_height, img_width)):
    nb_samples = nb_img_per_class*nb_classes

    # X_train = np.zeros((nb_samples, 3, img_width, img_height), dtype="uint8")
    from scipy import ndimage
    from scipy.misc import imresize
    im_list = []
    for img_name in sorted(os.listdir(path+os.sep+'home_office')):
        img = ndimage.imread(path+os.sep+'home_office'+os.sep+img_name)
        img_resized_transposed = imresize(img, (img_height, img_width)).transpose(2,0,1)
        im_list.append(img_resized_transposed)
    for img_name in sorted(os.listdir(path+os.sep+'office')):
        img = ndimage.imread(path+os.sep+'office'+os.sep+img_name)
        img_resized_transposed = imresize(img, (img_height, img_width)).transpose(2,0,1)
        im_list.append(img_resized_transposed)
    X_samples = np.asarray(im_list).astype("uint8")
    Y_samples = np.array([1,0]*nb_img_per_class+[0,1]*nb_img_per_class, dtype="uint8").reshape(nb_samples, nb_classes)

    # shuffle with the same random seed for both data and label
    np.random.seed(seed=123)
    np.random.shuffle(X_samples)
    np.random.seed(seed=123)
    np.random.shuffle(Y_samples)
    # and split
    test_ratio = 0.1
    nb_train_samples = int(nb_samples*(1-test_ratio))
    X_test = X_samples[nb_train_samples:, :, :, :]
    Y_test = Y_samples[nb_train_samples:, :]
    X_train = X_samples[:nb_train_samples, :, :, :]
    Y_train = Y_samples[:nb_train_samples, :]

    # # shuffle and split in one step,
    # from sklearn.model_selection import train_test_split    # split arrays into random train and test subsets
    # X_train, X_test, Y_train, Y_test = train_test_split(X_samples, Y_samples, test_size=0.1) # test_size 0.25 by default

    if K.image_dim_ordering() == 'tf':
        X_train = X_train.transpose(0, 2, 3, 1)
        X_test = X_test.transpose(0, 2, 3, 1)

    return (X_train, Y_train), (X_test, Y_test)

rel_path = 'datasets/data_256_HomeOrOff/test'   # test set is small with only 1000 images for each class
(X_train, Y_train), (X_test, Y_test) = load_data(path=os.getcwd()+os.sep+rel_path)
print('X_train shape: ', X_train.shape)
print(X_train.shape[0], ' train samples')
print(X_test.shape[0], ' test samples')

# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)

# set vgg-16 with fc layers
img_size = (3, img_width, img_height)
input_tensor = Input(batch_shape=(None,) + img_size)
model_vgg = vgg16.VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)

# add one Flatten and 3 Dense layers on the top, to overfit
base_model_output = model_vgg.output
base_model_output = Flatten()(base_model_output)
base_model_output = Dense(1024, activation='relu')(base_model_output)
base_model_output = Dropout(0.5)(base_model_output)
# # try to delete one Dense layer to check whether overfitting can be achieved
# base_model_output = Dense(1024, activation='relu')(base_model_output)
# base_model_output = Dropout(0.5)(base_model_output)
preds = Dense(2, activation='softmax')(base_model_output)
model_stacked = Model(model_vgg.input, preds)

nb_frozen_layers = 11
for layer in model_stacked.layers[:nb_frozen_layers]:
    layer.trainable = False
# 19 - train top fc layers only
# 15 - train block5 and fc layers
# 11 - train block5,4 and fc layers

learning_rate = 1e-4
model_stacked.compile(loss='categorical_crossentropy',
                      optimizer=SGD(lr=learning_rate, momentum=0.9),   # for fine tuning
                      metrics=['accuracy'])

# pre-processing
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

if not data_augmentation:
    print('No data augmentation.')
    model_stacked.fit(X_train, Y_train,
                      batch_size = batch_size,
                      nb_epoch= nb_epoch,
                      validation_data=(X_test, Y_test),
                      shuffle=True)
else:
    print('Using real-time data augmentation.')
    # apply ImageDataGenerator
    datagen = ImageDataGenerator(featurewise_center=False,  # True: set input mean to 0 over the dataset
                                 samplewise_center=False,   # True: set each sample mean to 0
                                 featurewise_std_normalization=False,   # True: divide inputs by std of dataset
                                 samplewise_std_normalization=False,    # True: divide each input by its std
                                 zca_whitening=False,       # no obvious improvement acc. to ZH
                                 rotation_range=0,          # randomly rotate in the range [0,180]
                                 width_shift_range=0.1,     # randomly shift horizontally
                                 height_shift_range=0.1,    # randomly shift vertically
                                 horizontal_flip=True,
                                 vertical_flip=False)
    # compute quantities required for feature-wise normalization
    datagen.fit(X_train)
    model_stacked.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                                samples_per_epoch=X_train.shape[0],
                                nb_epoch=nb_epoch,
                                validation_data=(X_test, Y_test))   # nothing change from generator to test data