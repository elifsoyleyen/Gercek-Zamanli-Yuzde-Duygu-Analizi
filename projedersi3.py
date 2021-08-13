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

new_model=tf.keras.models.load_model('proje2_modelcnnsparse.h5')

#resim=cv2.imread("happyboy.jpg")

path="haarcascade_frontalface_default.xml"
font_scale=1.5
font=cv2.FONT_HERSHEY_PLAIN

rectangle_bgr=(255,255,255)
img=np.zeros((500,500))
text="Some text in a box"
(text_width,text_height)=cv2.getTextSize(text,font,fontScale=font_scale,thickness=1)[0]
text_offset_x=10
text_offset_y=img.shape[0]-25

box_coords=((text_offset_x,text_offset_y),(text_offset_x+text_width+2,text_offset_y-text_height-2))

cv2.rectangle(img,box_coords[0],box_coords[1],rectangle_bgr,cv2.FILLED)
cv2.putText(img,text,(text_offset_x,text_offset_y),font,fontScale=font_scale,color=(0,0,0),thickness=1)

cap=cv2.VideoCapture(1)

if not cap.isOpened():
    cap=cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret,frame=cap.read()
    faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray,1.1,4)
    for x,y,w,h in faces:
        roi_gray=gray[y:y+h ,x:x+w]
        roi_color=frame[y:y+h ,x:x+w]
        cv2.rectangle(frame,(x,y),(x+w, y+h),(255,0,0),2)
        facess = faceCascade.detectMultiScale(roi_gray)
        if len(facess)==0:
            print("Yüz Bulunamadı!")
        else:
            for(ex,ey,ew,eh) in facess:
                face_roi=roi_color[ey:ey+eh, ex:ex+ew]

    final_image=cv2.resize(face_roi,(48,48))
    final_image=np.expand_dims(final_image,axis=0)
    final_image=final_image/255.0
    
    
    font=cv2.FONT_HERSHEY_SIMPLEX
    
    Predictions=new_model.predict(final_image)
    font_scale=1.5
    font=cv2.FONT_HERSHEY_PLAIN
    
    if(np.argmax(Predictions)==0):
        status="Sinirli"
        
        x1,y1,w1,h1=0,0,175,75    
        cv2.rectangle(frame,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)    
        cv2.putText(frame,status, (x1+ int(w1/10),y1 + int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)   
        cv2.putText(frame,status, (100,150),font,3,(0,0,255),2,cv2.LINE_4)    
        cv2.rectangle(frame ,(x,y),(x+w,y+h),(0,0,255))
        
    
    elif(np.argmax(Predictions)==1):
        status="Korku"
        
        x1,y1,w1,h1=0,0,175,75    
        cv2.rectangle(frame,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)    
        cv2.putText(frame,status, (x1+ int(w1/10),y1 + int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)   
        cv2.putText(frame,status, (100,150),font,3,(0,0,255),2,cv2.LINE_4)    
        cv2.rectangle(frame ,(x,y),(x+w,y+h),(0,0,255))    
    
    elif(np.argmax(Predictions)==2):
        status="Mutlu"
        
        x1,y1,w1,h1=0,0,175,75    
        cv2.rectangle(frame,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)    
        cv2.putText(frame,status, (x1+ int(w1/10),y1 + int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)   
        cv2.putText(frame,status, (100,150),font,3,(0,0,255),2,cv2.LINE_4)    
        cv2.rectangle(frame ,(x,y),(x+w,y+h),(0,0,255)) 
        
    elif(np.argmax(Predictions)==3):
        status="Dogal"
        
        x1,y1,w1,h1=0,0,175,75    
        cv2.rectangle(frame,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)    
        cv2.putText(frame,status, (x1+ int(w1/10),y1 + int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)   
        cv2.putText(frame,status, (100,150),font,3,(0,0,255),2,cv2.LINE_4)    
        cv2.rectangle(frame ,(x,y),(x+w,y+h),(0,0,255))  
    
    
    elif(np.argmax(Predictions)==4):
        status="Uzgun"
        
        x1,y1,w1,h1=0,0,175,75    
        cv2.rectangle(frame,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)    
        cv2.putText(frame,status, (x1+ int(w1/10),y1 + int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)   
        cv2.putText(frame,status, (100,150),font,3,(0,0,255),2,cv2.LINE_4)    
        cv2.rectangle(frame ,(x,y),(x+w,y+h),(0,0,255))  
    
    elif(np.argmax(Predictions)==5):
        status="Sasirma"
        
        x1,y1,w1,h1=0,0,175,75    
        cv2.rectangle(frame,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)    
        cv2.putText(frame,status, (x1+ int(w1/10),y1 + int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)   
        cv2.putText(frame,status, (100,150),font,3,(0,0,255),2,cv2.LINE_4)    
        cv2.rectangle(frame ,(x,y),(x+w,y+h),(0,0,255))  
    
    cv2.imshow('Yuzde Duygu Analizi',frame)

    if(cv2.waitKey(2) & 0xFF == ord('q')):
        break
  
cap.release()
cv2.destroyAllWindows()

































































        
        