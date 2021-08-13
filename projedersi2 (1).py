# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 13:18:45 2021

@author: elif
"""

import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt 
import numpy as np
import random
from tensorflow import keras 
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

print("aa")
directory="training/"   #traindataset
classes=['0','1','2','3','4','5','6']
train_data=[]
for kategori in classes: 
    path=os.path.join(directory,kategori)
    class_num=classes.index(kategori)
    for img in os.listdir(path):
        img_array=cv2.imread(path+"/"+img)
        train_data.append([img_array,class_num])
      
random.shuffle(train_data)
X=[] #data
y=[] #label
for features,label in train_data:
    X.append(features)
    y.append(label)

X=np.array(X).reshape(-1,224,224,3)
print(X.shape)
Y=np.array(y)
print(Y.shape)

X=X/255.0

X_train, X_test, y_train, y_test=train_test_split(X,Y, test_size=0.2,random_state=0)

model=tf.keras.applications.MobileNetV2()

base_input = model.layers[0].input
base_output = model.layers[-2].output

final_output = layers.Dense(128)(base_output)
final_ouput = layers.Activation('relu')(final_output)
final_output = layers.Dense(64)(final_ouput)
final_ouput = layers.Activation('relu')(final_output)
final_output = layers.Dense(7,activation='softmax')(final_ouput)

new_model = keras.Model(inputs=base_input,outputs=final_output)

new_model.summary()

new_model.compile(loss="sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"] )

new_model.fit(X_train,
              y_train,
              validation_data=(X_test,y_test),
              batch_size=16,
              epochs=20
              )

new_model.save('proje2_model.h5')



        
        