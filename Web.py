import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
from sklearn.preprocessing import LabelEncoder

class_labels = ['COPD', 'Healthy']

labelencoder = LabelEncoder()
labelencoder.fit(class_labels)

# Load your audio classifier model
model = tf.keras.models.load_model('audio_classification.hdf5')

st.title("Audio Classification")

uploaded_file = st.file_uploader("Upload Audio File", type=["wav"])

if uploaded_file is not None:
    # Load the selected audio file
    audio, sample_rate = librosa.load(uploaded_file, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

    mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)

    predicted_label = np.argmax(model.predict(mfccs_scaled_features), axis=-1)
    predicted_class = labelencoder.inverse_transform(predicted_label)[0]

    st.write(f"Predicted Class: {predicted_class}")
