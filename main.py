import sys
import subprocess

try:
    import cv2
except ModuleNotFoundError:
    print("Installing OpenCV...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
    import cv2 
import streamlit as st
import cv2
import os
from keras.models import load_model
from keras.utils import img_to_array,load_img
import numpy as np
st.set_page_config(page_title='Face Mask Detection',page_icon='https://5.imimg.com/data5/PI/FD/NK/SELLER-5866466/images-500x500.jpg')
# Check if files exist
if not os.path.exists("mask1.h5"):
    st.error("Error: mask.h5 not found! Please retrain the model.")
    st.stop()
if not os.path.exists("face.xml"):
    st.error("Error: face.xml not found!")
    st.stop()

facemodel=cv2.CascadeClassifier("face.xml")
maskmodel=load_model("mask1.h5", compile=False)
st.title('Welcome to FaceMask Detection 😷')
s=st.sidebar.selectbox('Home',('Home','IMAGE','WEB CAM'))
st.sidebar.image('https://cdn.hackernoon.com/images/oO6rUouOWRYlzw88QM9pb0KyMIJ3-bxfy3m27.png')
if(s=='Home'):
    st.image('https://cdn.dribbble.com/users/1815739/screenshots/12127262/dribbble_facemask_v1.gif')
elif(s=='IMAGE'):
     st.markdown('<center><h2>IMAGE DETECTION </h2></center>',unsafe_allow_html=True)
     file=st.file_uploader("Upload an Image")
     if file:
         b=file.getvalue()
         a=np.frombuffer(b,np.uint8)
         img=cv2.imdecode(a,cv2.IMREAD_COLOR)
         face=facemodel.detectMultiScale(img)
         for (x,y,l,w) in face:
             cv2.imwrite("temp.jpg",img[y:y+w,x:x+l])
             face_img=load_img("temp.jpg",target_size=(150,150,3))
             face_img=img_to_array(face_img)
             face_img=np.expand_dims(face_img,axis=0)
             pred=maskmodel.predict(face_img)[0][0]
             if(pred==1):
                 cv2.rectangle(img,(x,y),(x+l,y+w),(0,0,255),8)
             else:
                 cv2.rectangle(img,(x,y),(x+l,y+w),(0,255,0),8)
         st.image(img,channels='BGR')

elif s == 'WEB CAM':
    cam_index = st.text_input("Enter 0 for Primary Camera or 1 for Secondary Camera", "0")
    btn = st.button('Start Camera')

    if btn:
        k = int(cam_index)
        vid = cv2.VideoCapture(k)
        window = st.empty()  # Placeholder for updating frames
        
        stop_btn = st.button('Stop Camera')

        while vid.isOpened():
            flag, frame = vid.read()
            if not flag:
                break  # Stop if no frame is read

            faces = facemodel.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (150, 150))
                face_img = img_to_array(face_img) / 255.0  # Normalize
                face_img = np.expand_dims(face_img, axis=0)

                pred = maskmodel.predict(face_img)[0][0]
                color = (0, 0,255) if pred < 0.5 else (0, 255,0)  # Green = Mask, Red = No Mask
                label = "No Mask" if pred < 0.5 else " Mask"

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            window.image(frame, channels='BGR')  

            if stop_btn:
                vid.release()
                st.experimental_rerun()  

        vid.release()

