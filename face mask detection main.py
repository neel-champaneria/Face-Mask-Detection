# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 14:56:30 2020

@author: LENOVO
"""
import tensorflow
import keras
print(tensorflow.__version__)
print(keras.__version__)
print(tensorflow.keras.__version__)

# import necessary libraries
from tensorflow.keras.models import load_model
import cv2
import numpy as np
# import tkinter
# from tkinter import messagebox
 
import smtplib
from email.mime.multipart import MIMEMultipart 
from email.mime.text import MIMEText 
from email.mime.base import MIMEBase 
from email import encoders
 
# initialize tkinter
# root = tkinter.Tk()
# root.withdraw()
 
# load trained deep learning model
model = load_model("fmd_tf_2-1-0_2-2-4-tf.h5")

# classifier to detect face
face_det_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
 
# capture video
vid_source = cv2.VideoCapture(0)
 
 
# dictionary containing details of wearing mask and color of rectangle around face.
# if wearing mask then color would be green.
# if not color of rectangle around face would be red.
 
text_dict = {0:'Mask On', 1:'No Mask'}
rect_color_dict = {0:(0,255,0), 1:(255,0,0)}
 
# subject = 'Person without wearing the face mask detected'
# text = 'Alert, One visitor violeted the face mask policy. See in camera to recognize the user'
 
# while loop to continuosly detect camera feed
while(True):
  _, img = vid_source.read()
  grayscale_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  faces = face_det_classifier.detectMultiScale(grayscale_img, 1.3, 5)
 
  for (x, y, w, h) in faces:
    face_img = grayscale_img[y:y+h, x:x+w]
    resized_img = cv2.resize(face_img, (112,112))
    normalized_img = resized_img/255.0
    reshaped_img = np.reshape(normalized_img, (1,112,112,1))
    result = model.predict(reshaped_img)
 
    label = np.argmax(result, axis=1)[0]
 
    cv2.rectangle(img, (x, y), (x+w, y+h), rect_color_dict[label], 2)
    cv2.rectangle(img, (x, y-40), (x+w, y), rect_color_dict[label], -1)
    cv2.putText(img, text_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
 
    if (label == 1):
      # throw a warning message to tell user to wear a mask if not wearing one
      # this will stay open and no access will be given utill person wears a mask
      # messagebox.showwarning("Warning", "Access Denied. Please wear a face mask")
 
      #retpic, imgpic = vid_source.read()
      cv2.imwrite("newPicture.jpg", img)
      
      fromadd = "" # Email Address from where you want to send email
      password = "" # Password a fromadd email id
      toadd = "" # Email Address to whom you want to send email 
 
      msg = MIMEMultipart()
      msg['From'] = fromadd
      msg['to'] = toadd
      msg['subject'] = "Person without wearing the face mask detected"
      body = "Alert, One visitor violeted the face mask policy. See in camera to recognize the user"
      msg.attach(MIMEText(body, 'plain'))
      attachment = open("newPicture.jpg", 'rb')
      p = MIMEBase('application', 'octet-strem')
      p.set_payload((attachment).read())
      encoders.encode_base64(p)
      filename = "newPicture.jpg"
      p.add_header('Content-Disposition', "attachment; filename= %s" % filename)
      msg.attach(p)
      s = smtplib.SMTP('smtp.gmail.com', 587)
      s.starttls() 
      s.login(fromadd, password) 
      text = msg.as_string() 
      s.sendmail(fromadd, toadd, text) 
      s.quit()
    
    else:
      pass
      break
 
  cv2.imshow("Live Video Feed", img)
  key=cv2.waitKey(1)
 
  if(key==27):
    break
 
cv2.destroyAllWindows()
vid_source.release()