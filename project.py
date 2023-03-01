import cv2
import streamlit as st
from PIL import Image
import numpy as np

def detect_faces(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(np.array(image), (x, y), (x+w, y+h), (0, 255, 0), 2)
    return image

def main():
    st.set_page_config(page_title="Real Time Face Detection", page_icon=":smiley:", layout="wide")
    st.title("Real Time Face Detection App")
    st.markdown("Upload an image and detect faces in real-time")
    uploaded_image = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
        st.subheader("Processed Image")
        result_image = detect_faces(image)
        st.image(result_image, use_column_width=True)

if __name__ == '__main__':
    main()
