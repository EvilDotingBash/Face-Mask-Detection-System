import cv2
from keras.models import load_model
from keras.utils import img_to_array
import numpy as np
import os

# Ensure the model exists before loading
if not os.path.exists("mask1.h5"):
    raise FileNotFoundError("Error: mask1.h5 not found! Please retrain the model.")
if not os.path.exists("face.xml"):
    raise FileNotFoundError("Error: face.xml not found!")

# Load face detection model
facemodel = cv2.CascadeClassifier("face.xml")

# Load trained face mask detection model
try:
    maskmodel = load_model("mask1.h5", compile=False)
    print("Model loaded successfully!")
except Exception as e:
    print("Failed to load mask1.h5:", e)
    raise

# Open video file
vid = cv2.VideoCapture("mask2.mp4")

while vid.isOpened():
    flag, frame = vid.read()
    if not flag:
        break  # Exit if video ends or there's an issue reading the frame

    faces = facemodel.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]  # Crop face from frame
        face_img = cv2.resize(face_img, (150, 150))  # Resize for model input
        face_img = img_to_array(face_img)
        face_img = np.expand_dims(face_img, axis=0)  # Add batch dimension

        pred = maskmodel.predict(face_img)[0][0]
        color = (0, 255, 0) if pred == 1 else (0, 0, 255)  # Green = mask, Red = no mask
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 8)

    cv2.imshow("Face Mask Detection", frame)

    # Press 'x' to exit
    if cv2.waitKey(15) & 0xFF == ord('x'):
        break

vid.release()
cv2.destroyAllWindows()

