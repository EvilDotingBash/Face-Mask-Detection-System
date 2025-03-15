''' import cv2
import streamlit as st
import os
from keras.models import load_model
from keras.utils import img_to_array, load_img
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Set Page Config
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷"
)

# Check if model files exist
if not os.path.exists("mask1.h5") or not os.path.exists("face.xml"):
    st.error("Model files missing! Ensure 'mask1.h5' and 'face.xml' exist.")
    st.stop()

# Load models
facemodel = cv2.CascadeClassifier("face.xml")
maskmodel = load_model("mask1.h5", compile=False)

st.title("Welcome to FaceMask Detection 😷")

# Sidebar Navigation
s = st.sidebar.selectbox("Choose Mode", ("Home", "IMAGE", "WEB CAM"))
st.sidebar.image("https://cdn.hackernoon.com/images/oO6rUouOWRYlzw88QM9pb0KyMIJ3-bxfy3m27.png")

if s == "Home":
    st.image("https://cdn.dribbble.com/users/1815739/screenshots/12127262/dribbble_facemask_v1.gif")

elif s == "IMAGE":
     st.markdown("<h2 style='text-align: center;'>IMAGE DETECTION</h2>", unsafe_allow_html=True)
     file = st.file_uploader("Upload an Image")

     if file:
         img = cv2.imdecode(np.frombuffer(file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
         faces = facemodel.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

         for (x, y, w, h) in faces:
             face_img = cv2.resize(img[y:y+h, x:x+w], (150, 150))
             face_img = img_to_array(face_img) / 255.0
             face_img = np.expand_dims(face_img, axis=0)

             pred = maskmodel.predict(face_img)[0][0]
             color = (0, 255, 0) if pred > 0.5 else (0, 0, 255)
             label = "Mask" if pred > 0.5 else "No Mask"

             cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
             cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

         st.image(img, channels="BGR")

elif s == "WEB CAM":
    st.markdown("<h2 style='text-align: center;'>Live Face Mask Detection</h2>", unsafe_allow_html=True)

    class FaceMaskTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            faces = facemodel.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

            for (x, y, w, h) in faces:
                face_img = img[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (150, 150))
                face_img = img_to_array(face_img) / 255.0
                face_img = np.expand_dims(face_img, axis=0)

                pred = maskmodel.predict(face_img)[0][0]
                color = (0, 255, 0) if pred > 0.5 else (0, 0, 255)
                label = "Mask" if pred > 0.5 else "No Mask"

                cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="face-mask",
        video_transformer_factory=FaceMaskTransformer,
        media_stream_constraints={"video": True, "audio": False}  # Ensures webcam is properly accessed
    )'''
import cv2
import streamlit as st
import os
from keras.models import load_model
from keras.utils import img_to_array
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Set Page Config
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷"
)

# Check if model files exist
if not os.path.exists("mask1.h5") or not os.path.exists("face.xml"):
    st.error("Model files missing! Ensure 'mask1.h5' and 'face.xml' exist.")
    st.stop()

# Load models
facemodel = cv2.CascadeClassifier("face.xml")
maskmodel = load_model("mask1.h5", compile=False)

st.title("Welcome to FaceMask Detection 😷")

# Sidebar Navigation
s = st.sidebar.selectbox("Choose Mode", ("Home", "IMAGE", "WEB CAM"))
st.sidebar.image("https://cdn.hackernoon.com/images/oO6rUouOWRYlzw88QM9pb0KyMIJ3-bxfy3m27.png")

if s == "Home":
    st.image("https://cdn.dribbble.com/users/1815739/screenshots/12127262/dribbble_facemask_v1.gif")

elif s == "IMAGE":
    st.markdown("<h2 style='text-align: center;'>IMAGE DETECTION</h2>", unsafe_allow_html=True)
    file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    if file:
        img = cv2.imdecode(np.frombuffer(file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        faces = facemodel.detectMultiScale(img, scaleFactor=1.2, minNeighbors=6, minSize=(50, 50))

        if len(faces) == 0:
            st.warning("No faces detected. Try another image.")

        for (x, y, w, h) in faces:
            face_img = cv2.resize(img[y:y+h, x:x+w], (150, 150))
            face_img = img_to_array(face_img) / 255.0
            face_img = np.expand_dims(face_img, axis=0)

            pred = maskmodel.predict(face_img)[0][0]
            color = (0, 255, 0) if pred > 0.5 else (0, 0, 255)
            label = "Mask" if pred > 0.5 else "No Mask"

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        st.image(img, channels="BGR")

elif s == "WEB CAM":
    st.markdown("<h2 style='text-align: center;'>Live Face Mask Detection</h2>", unsafe_allow_html=True)

    class FaceMaskTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            faces = facemodel.detectMultiScale(img, scaleFactor=1.2, minNeighbors=6, minSize=(50, 50))

            if len(faces) == 0:
                print("No faces detected.")

            for (x, y, w, h) in faces:
                face_img = img[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (150, 150))
                face_img = img_to_array(face_img) / 255.0
                face_img = np.expand_dims(face_img, axis=0)

                pred = maskmodel.predict(face_img)[0][0]
                color = (0, 255, 0) if pred > 0.5 else (0, 0, 255)
                label = "Mask" if pred > 0.5 else "No Mask"

                cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="face-mask",
        video_transformer_factory=FaceMaskTransformer,
        media_stream_constraints={"video": True, "audio": False}
    )





