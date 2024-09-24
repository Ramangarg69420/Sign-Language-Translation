#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2  # This will now use the headless version
import threading
import numpy as np
import mediapipe as mp
from keras.models import load_model
import os
import speech_recognition as sr
import sounddevice as sd
import wave
from moviepy.editor import VideoFileClip, concatenate_videoclips
import pandas as pd
import time
from gtts import gTTS

# Load the Excel file
excel_path = "C:/Users/Raman/OneDrive/Desktop/Signlanguage/signlanguagedataset.xlsx"
df = pd.read_excel(excel_path)

# Create the dictionary that maps words to video files
sign_language_dict = dict(zip(df['Word'], df['VideoFilePath']))

# Function to standardize video properties
def standardize_videos(video_files):
    standardized_clips = []
    for video_file in video_files:
        if not os.path.exists(video_file):
            st.error(f"File not found: {video_file}")
            continue
        try:
            clip = VideoFileClip(video_file)
            standardized_clip = clip.resize((640, 480)).set_fps(30)
            standardized_clips.append(standardized_clip)
        except Exception as e:
            st.error(f"Error processing file {video_file}: {e}")
    return standardized_clips

# Function to concatenate videos
def concatenate_videos(video_files, output_path):
    clips = standardize_videos(video_files)
    if not clips:
        st.error("No valid video clips found.")
        return
    try:
        final_clip = concatenate_videoclips(clips)
        final_clip.write_videofile(output_path, codec="libx264", logger=None)
    except Exception as e:
        st.error(f"Error concatenating videos: {e}")

# Function to display the concatenated video
def display_concatenated_video(words):
    video_files = [sign_language_dict.get(word) for word in words if word in sign_language_dict]
    
    if video_files:
        output_path = "concatenated_video.mp4"
        concatenate_videos(video_files, output_path)
        if os.path.exists(output_path):
            video_bytes = open(output_path, 'rb').read()
            st.video(video_bytes)
        else:
            st.error("Failed to create the concatenated video.")
    else:
        st.write("No signs available for the given words.")

# Function to record audio using sounddevice
def record_audio(filename, duration=5, fs=44100):
    st.info("Recording...")

    try:
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()  # Wait until the recording is finished
        st.success("Recording complete")

        # Save the recorded audio to a file
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(fs)
            wf.writeframes(recording.tobytes())

    except Exception as e:
        st.error(f"Error during recording: {e}")

# Function to recognize speech from the recorded audio file
def recognize_speech_from_file(audio_file):
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)
        try:
            text = r.recognize_google(audio)
            with open("recognized_text.txt", "w") as f:
                f.write(text)
            st.success(f"Recognized text: {text}")
        except sr.UnknownValueError:
            st.error("Could not understand audio")
        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")

# Load sign language detection model
model_path = "C:/Users/Raman/OneDrive/Desktop/Signlanguage/sign to text data/modelss.keras"
file_path = "C:/Users/Raman/OneDrive/Desktop/Signlanguage/sign to text data/labelss.npy"
model = load_model(model_path)
label = np.load(file_path)
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Initialize session state
if "_components_callbacks" not in st.session_state:
    st.session_state["_components_callbacks"] = {}

# Streamlit app
st.title("Sign Language Conversion Web")

# Add a sidebar with options to choose between "Sign to Text" and "Text to Sign"
option = st.sidebar.selectbox("Choose an option:", ["Sign to Text", "Text to Sign"])

# Depending on the chosen option, display the corresponding content
if option == "Text to Sign":
    st.header("Text to Sign Language Converter")
    st.header("Or use voice input:")
    if st.button("Record Voice"):
        audio_file = "recorded_audio.wav"
        record_audio(audio_file)
        recognize_speech_from_file(audio_file)
        # Read the recognized text from the file
        if os.path.exists("recognized_text.txt"):
            with open("recognized_text.txt", "r") as f:
                recognized_text = f.read()
                st.text(f"Recognized text: {recognized_text}")
                words = recognized_text.lower().split()
                display_concatenated_video(words)
    # Input text from the user
    user_input = st.text_input("Enter text:")
    if st.button("Convert Text to Sign Language"):
        # Process the input and display the corresponding signs
        if user_input:
            words = user_input.lower().split()
            display_concatenated_video(words)
    
elif option == "Sign to Text":
    st.header("Sign to Text Conversion")
    st.write("Use your camera to display sign language and convert it to text.")
    
    class Mooddetector:
        def __init__(self):
            self.last_prediction_time = time.time()
            self.prediction_interval = 5  # Time interval in seconds
            self.last_prediction = None
            self.words = []  # To store words
            self.output_file = "C:/Users/Raman/OneDrive/Desktop/Signlanguage/recognized_sentence.txt"
            self.audio_file = "C:/Users/Raman/OneDrive/Desktop/Signlanguage/recognized_sentence_audio.mp3"
            self.lock = threading.Lock()  # Lock for file writing to avoid race conditions

        def open_file(self):
            # Automatically opening the recognized sentence file
            try:
                if os.name == 'nt':
                    os.startfile(self.output_file)
            except Exception as e:
                st.error(f"Error opening file: {e}")

        def convert_text_to_speech(self):
            # Convert the recognized sentence to speech and save as an audio file
            try:
                with open(self.output_file, "r") as f:
                    text = f.read().strip()
                    if text:
                        tts = gTTS(text=text, lang='en')
                        tts.save(self.audio_file)
                        st.success("Audio generated successfully.")
                        st.audio(self.audio_file)
                    else:
                        st.error("No text found in the file")
            except Exception as e:
                st.error(f"Error generating audio: {e}")

        def process_prediction(self, prediction):
            # Process the prediction in a separate thread to avoid blocking
            with self.lock:
                if prediction:
                    self.words.append(prediction)
                    with open(self.output_file, "a") as f:
                        f.write(prediction + " ")

                    # Generate speech in a separate thread to avoid blocking
                    threading.Thread(target=self.convert_text_to_speech).start()

        def recv(self, frame):
            current_time = time.time()
            frm = frame.to_ndarray(format="bgr24")
            frm = cv2.flip(frm, 1)
            res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
            lst = []

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for _ in range(42):
                    lst.append(0.0)
            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for _ in range(42):
                    lst.append(0.0)
            lst = np.array(lst).reshape(1, -1)

            if current_time - self.last_prediction_time >= self.prediction_interval:
                self.last_prediction = label[np.argmax(model.predict(lst))]
                self.last_prediction_time = current_time  # Reset the timer

                # Run prediction processing in a separate thread
                if self.last_prediction and self.last_prediction != "NO SIGN":
                    threading.Thread(target=self.process_prediction, args=(self.last_prediction,)).start()

            if self.last_prediction:
                cv2.putText(frm, self.last_prediction, (50, 50), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)
                np.save("mood.npy", np.array([self.last_prediction]))

            drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
            drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)
            return av.VideoFrame.from_ndarray(frm, format="bgr24")

    detector = Mooddetector()
    webrtc_streamer(key="example", video_frame_callback=detector.recv)

    if st.button("Open Recognized Sentence File"):
        detector.open_file()
    
    if st.button("Listen Recognized Sentence"):
        detector.convert_text_to_speech()  # Generate audio and save to file
      
        
        


# In[ ]:





# In[ ]:




