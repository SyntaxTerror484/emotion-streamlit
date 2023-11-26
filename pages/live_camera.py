from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
from keras.models import load_model
import numpy as np
from keras.utils import img_to_array
import streamlit as st

emotion_labels = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}
colors = {0:(0, 128, 255), 1:(246, 61, 252), 2:(246, 61, 252), 3:(0,255,90), 4:(204, 204, 204), 5:(237, 28, 63), 6:(252,238,33)}

cascade = cv2.CascadeClassifier('EmotionDetection/haarcascade_frontalface_default.xml')

@st.cache_resource
def load_required_models():
    prediction_model = load_model('EmotionDetection/model')
    age_model = load_model('EmotionDetection/age')

    return prediction_model, age_model

prediction_model, age_model = load_required_models()

class VideoProcessor:
    def recv(self, frame):

        frame_ = frame.to_ndarray(format="bgr24")


        return av.VideoFrame.from_ndarray(frame_, format='bgr24')

webrtc_streamer(key="daf", video_processor_factory=VideoProcessor)
