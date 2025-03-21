
import cv2 
import streamlit as st
import cv2
import os
from keras.models import load_model
from keras.utils import img_to_array,load_img
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

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

elif s == "WEB CAM":
    st.markdown("<h2 style='text-align: center;'>Live Face Mask Detection</h2>", unsafe_allow_html=True)

    class FaceMaskTransformer(VideoTransformerBase):
        def transform(self, frame):
            # img = frame.to_ndarray(format="bgr24")
            faces = facemodel.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (150, 150))
                face_img = img_to_array(face_img) / 255.0
                face_img = np.expand_dims(face_img, axis=0)

                pred = maskmodel.predict(face_img)[0][0]
                color = (0, 0,255) if pred < 0.5 else (0, 255,0)  # Green = Mask, Red = No Mask
                label = "No Mask" if pred < 0.5 else " Mask"

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            return av.VideoFrame.from_ndarray(frame, format="bgr24")

    webrtc_streamer(
        key="face-mask",
        video_transformer_factory=FaceMaskTransformer,
        media_stream_constraints={"video": True, "audio": False}
    )
        

