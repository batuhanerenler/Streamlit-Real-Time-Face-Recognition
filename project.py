import cv2
import streamlit as st
from mtcnn import MTCNN
from deepface import DeepFace

# set page title and icon
st.set_page_config(page_title="Face Detection App", page_icon=":guardsman:")


# function to detect faces using MTCNN
def detect_faces(image):
    detector = MTCNN()
    result = detector.detect_faces(image)
    if result != []:
        for person in result:
            bounding_box = person['box']
            keypoints = person['keypoints']
            cv2.rectangle(
                image,
                (bounding_box[0], bounding_box[1]),
                (bounding_box[0]+bounding_box[2], bounding_box[1]+bounding_box[3]),
                (0, 155, 255),
                2
            )
        return image, True
    else:
        return image, False


# function to get gender and age using DeepFace
def get_gender_age(image):
    try:
        result = DeepFace.analyze(image, actions=['gender', 'age'])
        gender = result['gender']
        age = result['age']
        return gender, age
    except:
        return None, None


# main function
def main():
    st.title("Face Detection App")

    # select image from local files
    image_file = st.file_uploader(
        "Upload an image to detect faces, gender, and age", 
        type=['jpg', 'jpeg', 'png']
    )

    if image_file is not None:
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
        st.image(image, channels="BGR", caption="Uploaded Image")

        # detect faces
        detected_image, face_detected = detect_faces(image)
        if face_detected:
            st.success("Faces detected!")
        else:
            st.warning("No faces detected.")

        # get gender and age
        gender, age = get_gender_age(detected_image)
        if gender is not None and age is not None:
            st.success("Gender and age detected!")
            st.write("Gender:", gender)
            st.write("Age:", age)
        else:
            st.warning("Could not detect gender and age.")
    else:
        st.warning("Upload an image to get started.")


if __name__ == "__main__":
    main()

    
