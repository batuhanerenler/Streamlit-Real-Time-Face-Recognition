import cv2
import streamlit as st

def detect_faces(video_capture):
    
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    
    while True:
        
        ret, frame = video_capture.read()

      
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def main():
    
    st.title('Real-time Face Recognition')

    
    video_capture = cv2.VideoCapture(0)

    
    video_display = st.image([])

    
    for frame in detect_faces(video_capture):
        video_display.image(frame)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    video_capture.release()


if __name__ == '__main__':
    main()
