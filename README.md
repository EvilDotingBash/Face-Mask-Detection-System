# Face Mask Detection System

## Overview
The **Face Mask Detection System** is a deep learning-based application that detects whether a person is wearing a face mask or not. It utilizes **Convolutional Neural Networks (CNNs)** for image classification and is trained on a dataset of masked and unmasked faces.

## Features
✅ Real-time face mask detection using a webcam or pre-recorded images.  
✅ Trained with a deep CNN model for high accuracy.  
✅ Utilizes OpenCV for image preprocessing and face detection.  
✅ Supports deployment on various platforms, including mobile and web applications.  
✅ Interactive visualization of predictions and confidence scores.

## Technologies Used
🧠 **Deep Learning Framework**: TensorFlow/Keras  
🐍 **Programming Language**: Python  
📷 **Image Processing**: OpenCV  
📂 **Dataset**: Publicly available face mask datasets  
🏗️ **Model Architecture**: Convolutional Neural Networks (CNNs)  
🖥️ **Backend**: Streamlit (for interactive UI)  

## Files Structure
- **cnn.py**: Contains the CNN model architecture and training pipeline.  
- **backend.py**: Manages the backend logic using Streamlit for UI and API integration.  
- **main.py**: The entry point to run the full system, integrating the model and backend.  

## Dataset
The dataset consists of images categorized into two classes:
1. 😷 **With Mask**
2. 😶 **Without Mask**

The images are preprocessed, augmented, and split into training and testing sets for optimal model performance.

## Installation
1️⃣ Clone the repository:
   ```sh
   git clone https://github.com/EvilDotingBash/Face-Mask-Detection-System.git
   cd Face-Mask-Detection-System
   ```
2️⃣ Install the required dependencies manually:
   ```sh
   pip install tensorflow opencv-python streamlit numpy pandas matplotlib
   ```
3️⃣ Download and prepare the dataset.

## Model Training
🚀 To train the model, run:
```sh
python cnn.py
```
This will train the CNN model using the specified dataset and save the trained model for inference.

## Running the Face Mask Detector
🎥 To detect masks in real-time using a webcam, execute:
```sh
python main.py
```

## Backend & UI
The system includes a **Streamlit** backend for an interactive user interface. Run the application with:
```sh
streamlit run backend.py
```
This will launch a web-based interface for face mask detection.

## Results
📊 **Achieved high accuracy** in mask detection.  
📈 The model generalizes well on unseen images.  
📌 Interactive visualization of confidence scores and bounding boxes for detected faces.

## Future Enhancements
✨ Improve model performance with more diverse datasets.  
📱 Deploy the model as a web or mobile application.  
⚡ Optimize inference speed for real-time applications.  
🎮 Add an interactive UI for better user experience.

## Deployment
🚀 The application is deployed and accessible at:
🔗 [Live Demo](https://dummy-deployment-link.com)

## Contributors
👩‍💻 **Your Name** (Add your details here)

## License
📜 This project is licensed under the MIT License - see the LICENSE file for details.

## Repository Link
🔗 [GitHub Repository](https://github.com/EvilDotingBash/Face-Mask-Detection-System)

