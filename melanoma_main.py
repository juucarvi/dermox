import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

#1Application heading
st.title("Dermoverse Skin Cancer Detector")

st.subheader("#MSCHOOLS")

url_link = "http://www.dermoverse.org"
url_text = "Dermoverse.org"
text = f"Visit us in [{url_text}]({url_link})."

st.markdown(text, unsafe_allow_html=True)

#st.markdown('Visit us in: **dermoverse.org**.')

#1Brief summary of what the application does
st.subheader("This BETA (BrUNO-1) can classify potential skin cancer images into two classes, whether they are benign or malignant. The images uploaded should be clinically made. ")
    
# Ask for the username
username = st.text_input("Enter your nickname:")

#1Information of what kind of images should be uploaded.
st.markdown("In the upcoming versions phone-made pictures will be supported.")

st.markdown("Note that is just a beta. Consult with a professional for further information.")

# Load the pre-trained model
model = tf.keras.models.load_model('dermodev.h5')

# Define the class labels
class_labels = ['Malignant', 'Benign']

# Set allowed file types
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

# Define the Streamlit app
uploaded_file = st.file_uploader("Choose an image...", type=ALLOWED_EXTENSIONS)

# Define the "Run Model" button
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded picture")
    run_model = st.button("Predict")
else:
    run_model = False

# Make a prediction on the uploaded image when the "Run Model" button is clicked
if run_model:
    image = Image.open(uploaded_file)
    image = image.resize((224, 224))
    image = np.array(image)
    image = image.astype('float32') / 255.0
    prediction = model.predict(np.expand_dims(image, axis=0))[0]
    percentages = tf.nn.softmax(prediction) * 100

    # Print the prediction
    st.write("Prediction:")
    for i, percentage in enumerate(percentages):
        class_label = class_labels[i]
        st.write(f"{class_label}: {percentage:.2f}%")
        
        
 # Save the prediction to a text file
    predicted_class = class_labels[prediction.argmax()]
    with open('prediction.txt', 'w') as f:
        f.write(predicted_class)
    
    # Add a button to download the prediction file
    download_button = st.download_button(
        label='Download prediction',
        data = f"Dermoverse Melanoma Detector\n   \nUsername: {username}\nMalignant: {percentages[0]:.2f}%\nBenign: {percentages[1]:.2f}%\n  \nCopyright ©2023",
        file_name='prediction.txt',
        mime='text/plain'
    )
