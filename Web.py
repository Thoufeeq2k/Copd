import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
from sklearn.preprocessing import LabelEncoder
from pydub import AudioSegment
import os

def convert_mp3_to_wav(mp3_data):
    print("Converting MP3 to WAV...")
    audio = AudioSegment.from_mp3(mp3_data)
    wav_file = mp3_data.name.replace('.mp3', '.wav')
    print("Exporting WAV file...")
    audio.export(wav_file, format="wav")
    print("Conversion complete.")
    return wav_file


class_labels = ['COPD', 'Healthy']

labelencoder = LabelEncoder()
labelencoder.fit(class_labels)

# Load your audio classifier model
model = tf.keras.models.load_model('audio_classification.hdf5')

st.title("Audio Classification")

uploaded_file = st.file_uploader("Upload Audio File", type=["mp3","wav"])

if uploaded_file is not None:
        if uploaded_file.type == "audio/wav":
            # No need for conversion, use the uploaded WAV file directly
            wav_file = uploaded_file
            st.success("Uploaded WAV file used directly.")
        else:
            # Convert MP3 to WAV
            wav_file = convert_mp3_to_wav(uploaded_file)
            st.success("MP3 file converted to WAV format.")

if uploaded_file is not None:
    # Load the selected audio file
    audio, sample_rate = librosa.load(uploaded_file, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)

    predicted_label = np.argmax(model.predict(mfccs_scaled_features), axis=-1)
    predicted_class = labelencoder.inverse_transform(predicted_label)[0]

    st.write(f"Predicted Class: {predicted_class}")
