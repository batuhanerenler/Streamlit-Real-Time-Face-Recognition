import cv2
import streamlit as st

def detect_faces(video_capture):
    # Load the cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Loop over frames from the video stream
    while True:
        # Read a frame from the video stream
        ret, frame = video_capture.read()

        # Check if the frame is empty
        if not ret:
            continue

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Encode the frame as JPEG and yield it to the client
        _, jpeg = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        yield jpeg.tobytes()

# Define the main function for the Streamlit app
def main():
    # Set the title of the app
    st.title('Real-time Face Recognition')

    # Open the video capture device (0 is usually the built-in webcam)
    video_capture = cv2.VideoCapture(0)

    # Create a display container for the video stream
    video_display = st.empty()

    # Loop over frames from the video stream and display the resulting frame
    for frame_bytes in st.websocket(detect_faces(video_capture)):
        video_display.image(frame_bytes)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture device and close the display container
    video_capture.release()

# Run the app
if __name__ == '__main__':
    main()
