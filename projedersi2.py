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

new_model=tf.keras.models.load_model('proje2_model.h5')

resim=cv2.imread("happyboy.jpg")

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
griresim = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(griresim,1.1,4)

for x,y,w,h in faces:
    roi_gray = griresim[y:y+h, x:x+w]
    roi_color = resim[y:y+h, x:x+w]
    cv2.rectangle(resim,(x,y),(x+w, y+h),(255,0,0),2)
    facess = faceCascade.detectMultiScale(roi_gray)
    if len(facess)==0:
        print("Yüz Bulunamadı!")
    else:
        for(ex,ey,ew,eh) in facess:
            face_roi=roi_color[ey:ey+eh, ex:ex+ew]

plt.imshow(cv2.cvtColor(resim, cv2.COLOR_BGR2RGB))            

final_image=cv2.resize(face_roi, (224,224))
final_image=np.expand_dims(final_image,axis=0)
final_image=final_image/255.0

predictions=new_model.predict(final_image)

print(predictions[0])    
    







        
        