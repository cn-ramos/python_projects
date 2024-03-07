import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import pdb
import pandas as pd
import seaborn as sns
import datetime, os
import warnings

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization
from tensorflow.keras.models import load_model
from keras.utils.generic_utils import get_custom_objects

from tensorflow.keras import backend as K
from PIL import Image, ImageOps



n_classes = 37
input_size = (64,64,3)
batch_size = 100


def downsize(image):
    '''
    Reads in image and eturns a cropped and compressed version of 'image'. 'image' must be a string path to image 
        (ex: image = 'img/training_images/xxxxxxx.jpg')

    -------------
    Output: Downsized Image type object (works with plt.imshow)
    '''
    
    im = Image.open('imgs/training_images/' + image + '.jpg')
    
    cropped_im = im.crop((75,75,349,349))
    newsize = (64,64)
    final_im = cropped_im.resize(newsize, resample = Image.Resampling.BILINEAR) #bilinear interpolation
    return tf.keras.utils.img_to_array(final_im)

class readin(tf.keras.utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size):
        self.image_filenames, self.labels = np.array(image_filenames), labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([downsize(file_name)
               for file_name in batch_x]), np.array(batch_y)
    
with tf.device('/gpu:0'):
    model = Sequential() #input layer

    model.add(Conv2D(32, (3, 3), input_shape=(64,64, 3))) #1st convolutional
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) #pooling layer

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(rate = 0.2))  

    
    model.add(Dense(n_classes))
    model.add(Activation('sigmoid'))

with tf.device('/gpu:0'):
    resnet = tf.keras.applications.ResNet50(
        include_top=True,
        weights='imagenet',
        input_tensor=tf.keras.Input(shape=(64,64,3)),
        input_shape=None,
        pooling="avg",
        classes=1000)
    res_model = Sequential()
    res_model.add(resnet)
    res_model.add(Flatten())
    res_model.add(Dense(n_classes))
    res_model.add(Activation('sigmoid'))
