import cv2
import streamlit as st
import pyv4l2

def detect_faces(video_capture):
    # Load the cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Loop over frames from the video stream
    while True:
        # Read a frame from the video stream
        frame = video_capture.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Convert the frame to RGB and yield it to the client
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Define the main function for the Streamlit app
def main():
    # Set the title of the app
    st.title('Real-time Face Recognition')

    # Open the video capture device (0 is usually the built-in webcam)
    video_capture = pyv4l2.Capture(0)

    # Create a placeholder image for the video stream
    video_display = st.empty()

    # Loop over frames from the video stream and display the resulting frame
    for frame in detect_faces(video_capture):
        video_display.image(frame, channels='RGB')

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture device
    video_capture.close()

# Run the app
if __name__ == '__main__':
    main()
