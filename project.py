import streamlit as st
import cv2
import numpy as np
import urllib.request
import os
import sys

# Load the pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Download and convert the pre-trained age and gender prediction models from the GitHub repository
age_url = "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel"
age_filename = "age_net.caffemodel"
deploy_age_url = "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/deploy_age.prototxt"
deploy_age_filename = "deploy_age.prototxt"
gender_url = "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel"
gender_filename = "gender_net.caffemodel"
deploy_gender_url = "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/deploy_gender.prototxt"
deploy_gender_filename = "deploy_gender.prototxt"

dir_path = os.path.dirname(os.path.realpath(__file__))

if not os.path.exists(os.path.join(dir_path, age_filename)):
    print("Downloading age prediction model...")
    urllib.request.urlretrieve(age_url, os.path.join(dir_path, age_filename))
if not os.path.exists(os.path.join(dir_path, deploy_age_filename)):
    print("Downloading age prediction model deploy file...")
    urllib.request.urlretrieve(deploy_age_url, os.path.join(dir_path, deploy_age_filename))
if not os.path.exists(os.path.join(dir_path, gender_filename)):
    print("Downloading gender prediction model...")
    urllib.request.urlretrieve(gender_url, os.path.join(dir_path, gender_filename))
if not os.path.exists(os.path.join(dir_path, deploy_gender_filename)):
    print("Downloading gender prediction model deploy file...")
    urllib.request.urlretrieve(deploy_gender_url, os.path.join(dir_path, deploy_gender_filename))

try:
    os.system(f"python {cv2.samples.findFile('tf_text_graph_ssd.py')} --input {deploy_age_filename} --output {os.path.splitext(age_filename)[0]}.caffemodel --transform-type caffe")
    age_filename = f"{os.path.splitext(age_filename)[0]}.caffemodel"
except:
    print("Error: could not convert age prediction model to Caffe format.")
    sys.exit(1)
try:
    os.system(f"python {cv2.samples.findFile('tf_text_graph_ssd.py')} --input {deploy_gender_filename} --output {os.path.splitext(gender_filename)[0]}.caffemodel --transform-type caffe")
    gender_filename = f"{os.path.splitext(gender_filename)[0]}.caffemodel"
except:
    print("Error: could not convert gender prediction model to Caffe format.")
    sys.exit(1)

# Load the pre-trained age and gender prediction models from OpenCV
age_net = cv2.dnn.readNet(cv2.samples.findFile(os.path.join(dir_path, age_filename)), cv2.samples.findFile(os.path.join(dir_path, deploy_age_filename)))
gender_net = cv2.dnn.readNet(cv2.samples.findFile(os.path.join(dir_path, gender_filename)), cv2.samples.findFile(os.path.join(dir_path, deploy_gender_filename)))

# Define a function to detect faces, mark them with a square, and predict age and gender
def detect_faces(image):
        # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop over each detected face
    for (x, y, w, h) in faces:
        # Mark the face with a green square
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Extract the face ROI (region of interest) and resize it to 227x227
        face_roi = image[y:y+h, x:x+w]
        face_blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # Predict the age and gender of the face ROI using the pre-trained models
        age_net.setInput(face_blob)
        age_preds = age_net.forward()
        age = int(age_preds[0].dot(np.arange(0, 101).reshape(101, 1)).flatten()[0])
        gender_net.setInput(face_blob)
        gender_preds = gender_net.forward()
        gender = "Male" if gender_preds[0][0] > gender_preds[0][1] else "Female"

        # Display the predicted age and gender on the annotated image
        label = f"{gender}, {age}"
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    return image

# Define the Streamlit app
def app():
    # Set the title and sidebar
    st.set_page_config(page_title="Real-time Face Recognition", page_icon=":guardsman:", layout="wide")
    st.sidebar.title("Real-time Face Recognition")
    st.sidebar.markdown("Upload an image to detect faces and predict age and gender.")

    # Upload an image using the file uploader widget
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # If an image was uploaded, display it and process it
    if uploaded_file is not None:
        # Load the image
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Detect faces, mark them with a square, and predict age and gender
        annotated_image = detect_faces(image)

        # Display the annotated image
        st.image(annotated_image, channels="BGR")

# Run the Streamlit app
if __name__ == "__main__":
    app()
