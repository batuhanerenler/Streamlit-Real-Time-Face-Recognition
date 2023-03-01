import streamlit as st
import cv2
import numpy as np
import os

# Load the pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load the pre-trained age and gender prediction models from OpenCV
dir_path = os.path.dirname(os.path.realpath(__file__))
age_net = cv2.dnn.readNet(cv2.samples.findFile(os.path.join(dir_path, "age_net.caffemodel")), cv2.samples.findFile(os.path.join(dir_path, "deploy_age.prototxt")))
gender_net = cv2.dnn.readNet(cv2.samples.findFile(os.path.join(dir_path, "gender_net.caffemodel")), cv2.samples.findFile(os.path.join(dir_path, "deploy_gender.prototxt")))

# Define a function to detect faces, mark them with a square, and predict age and gender
def detect_faces(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image using the face detection model
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through each detected face
    for (x, y, w, h) in faces:
        # Draw a square around the face on the original color image
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Extract the face ROI
        face_roi = image[y:y+h, x:x+w]

        # Preprocess the face ROI for age and gender prediction
        blob = cv2.dnn.blobFromImage(face_roi, scalefactor=1.0, size=(227, 227), mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=False, crop=False)

        # Predict the age of the face ROI using the age prediction model
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = int(age_preds[0].dot(list(range(0, 101))))

        # Predict the gender of the face ROI using the gender prediction model
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = "Male" if gender_preds[0][0] > gender_preds[0][1] else "Female"

        # Draw the predicted age and gender on the original color image
        cv2.putText(image, f"{age} years {gender}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # Return the annotated image
    return image

# Define the streamlit app
def app():
    # Set the title and description
    st.title("Face Detection and Age/Gender Prediction")
    st.markdown("Upload an image to detect faces and predict age and gender.")

    # Create a file uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # If an image was uploaded, display it and process it
    if uploaded_file is not None:
        # Load the image
        image = cv2.imdecode
    # If an image was uploaded, display it and process it
    if uploaded_file is not None:
        # Load the image
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Detect faces, mark them with a square, and predict age and gender
        annotated_image = detect_faces(image)

        # Display the annotated image
        st.image(annotated_image, channels="BGR")

# Run the streamlit app
if __name__ == "__main__":
    app()
