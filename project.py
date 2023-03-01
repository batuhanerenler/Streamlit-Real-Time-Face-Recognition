import streamlit as st
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

# Load the ViT model and image processor
model_name = 'google/vit-base-patch16-224'
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# Define the ImageNet class labels
with open('imagenet_classes.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Create the Streamlit app
st.title('ViT Image Classifier')

uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    inputs = processor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits.detach().numpy()[0]
    predicted_class_idx = logits.argmax()
    predicted_class = labels[predicted_class_idx]
    st.image(image, caption=f'Predicted class: {predicted_class}', use_column_width=True)
