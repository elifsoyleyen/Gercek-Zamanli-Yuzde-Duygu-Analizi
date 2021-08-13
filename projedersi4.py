# -*- coding: utf-8 -*-
"""
Created on Sun May 16 15:28:38 2021

@author: Sencer
"""
from keras.regularizers import l2
from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from tensorflow.keras import layers
from tensorflow import keras 
from keras.layers import Input, Lambda, Dense, Flatten, Activation
from keras.models import Model
from keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt 
import numpy as np
import random
from tensorflow import keras 
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from keras import applications
from keras.applications.resnet50 import ResNet50
from keras.optimizers import Adam
import datetime

NAME = "emotion_detection"
#
#IMAGE_SIZE = [224, 224]
#vgg = applications.InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
#
#for layer in vgg.layers:
#    layer.trainable = False
#    
#x = Flatten()(vgg.output)
#prediction = Dense(7, activation='softmax')(x)
#new_model = Model(inputs=vgg.input, outputs=prediction)
model=applications.MobileNetV2()

base_input = model.layers[0].input
base_output = model.layers[-2].output

final_output = Dense(128)(base_output)
final_ouput = Activation('relu')(final_output)
final_output = Dense(64)(final_ouput)
final_ouput = Activation('relu')(final_output)
final_output = Dense(6,activation='softmax')(final_ouput)

model = Model(inputs=base_input,outputs=final_output)

#model = Sequential()
#
#model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 3), data_format='channels_last', kernel_regularizer=l2(0.01)))
#model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
#model.add(BatchNormalization())
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#model.add(Dropout(0.5))
#
#model.add(Conv2D(2*64, kernel_size=(3, 3), activation='relu', padding='same'))
#model.add(BatchNormalization())
#model.add(Conv2D(2*64, kernel_size=(3, 3), activation='relu', padding='same'))
#model.add(BatchNormalization())
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#model.add(Dropout(0.5))
#
#model.add(Conv2D(2*2*64, kernel_size=(3, 3), activation='relu', padding='same'))
#model.add(BatchNormalization())
#model.add(Conv2D(2*2*64, kernel_size=(3, 3), activation='relu', padding='same'))
#model.add(BatchNormalization())
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#model.add(Dropout(0.5))
#
#model.add(Conv2D(2*2*2*64, kernel_size=(3, 3), activation='relu', padding='same'))
#model.add(BatchNormalization())
#model.add(Conv2D(2*2*2*64, kernel_size=(3, 3), activation='relu', padding='same'))
#model.add(BatchNormalization())
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#model.add(Dropout(0.5))
#
#model.add(Flatten())
#
#model.add(Dense(2*2*2*64, activation='relu'))
#model.add(Dropout(0.4))
#model.add(Dense(2*2*64, activation='relu'))
#model.add(Dropout(0.4))
#model.add(Dense(2*64, activation='relu'))
#model.add(Dropout(0.5))
#
#model.add(Dense(6, activation='softmax'))

model.summary()
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])
#new_model.compile(loss="categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"] )


data_generator = ImageDataGenerator(
                    rescale = 1. / 255,               
                    validation_split = 0.2)
            
train_gen=data_generator.flow_from_directory("./fer2013",
                    target_size=(224,224),
                    shuffle=True,
                    batch_size=32,
                    class_mode='sparse',
                    subset='training')                    
    
test_gen=data_generator.flow_from_directory("./fer2013",
                    target_size=(224,224),
                    shuffle=False,
                    batch_size=32,
                    class_mode='sparse',
                    subset='validation')
modelfit=model.fit(
            train_gen,
            validation_data=test_gen,
            epochs=50,
            steps_per_epoch=len(train_gen),
            validation_steps=len(test_gen)
            )

plt.figure(figsize=(10,2.5))
plt.subplot(1, 2, 1)
plt.plot(modelfit.history['accuracy'])
plt.plot(modelfit.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.figure(figsize=(10,2.5))
plt.subplot(1, 2, 1)
plt.plot(modelfit.history['loss'])
plt.plot(modelfit.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')


model.save('proje2_modelcnnsparse.h5')



        
        