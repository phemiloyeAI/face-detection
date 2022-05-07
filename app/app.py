"""Streamlit web app"""

import os
import tempfile
import cv2
import torch
import numpy as np
from PIL import Image
import streamlit as st
from retinaface.pre_trained_models import get_model
from retinaface.utils import vis_annotations

st.set_option("deprecation.showfileUploaderEncoding", False)

@st.cache
def cached_model():
    m = get_model("resnet50_2020-07-20", max_size=1048, device="cpu")
    m.eval()
    return m


model = cached_model()

st.title("Detect faces and key points")

menu = st.sidebar.selectbox("App mode", options=["Image", "Video"])

st.write("")
if menu == "Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg"])
    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))
        # st.image(image)
        st.write("Detecting faces...")
        with torch.no_grad():
            annotations = model.predict_jsons(image)
        if not annotations[0]["bbox"]:
            st.write("No faces detected")
        else:
            visualized_image = vis_annotations(image, annotations)
            st.write("Number of Detected Faces: ", len(annotations))

            st.image(visualized_image, use_column_width=True)

if menu == "Video":
    video_file = st.file_uploader("Choose a video", type=["mp4", "webm"])
    tfile = tempfile.NamedTemporaryFile(delete=False)
    button = st.button("Process Video")

    if video_file:
        if button:  
            tfile.write(video_file.read())

            cap = cv2.VideoCapture(tfile.name)
            st.write("Processing video...")
            st.write("\n")
            st.write("***This process may take a while to complete.***")

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            fourcc = cv2.VideoWriter_fourcc(*'VP08')

            out_path = os.path.join(os.getcwd(), "output.webm")
            writer = cv2.VideoWriter(out_path, fourcc, 20, (width, height))

            num_faces = 0
            i = 0   

            while True:
                ret, frame = cap.read()
                if not ret:
                    cv2.putText(visualized_image, f"Faces Counter: {str(int(num_faces/i))}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    writer.write(visualized_image)
                    st.write("Number of Detected Faces: ", str(int(num_faces/i)))
                    break
                else:
                    with torch.no_grad():
                        annotations = model.predict_jsons(frame, 0.05)
                    if not annotations[0]["bbox"]:
                        st.write("No faces detected")
                    else:
                        num_faces += len(annotations)
                        visualized_image = vis_annotations(frame, annotations)
                        writer.write(visualized_image)
                i += 1

            cap.release()
            writer.release()

            video_file = open(out_path, "rb")
            video_bytes = video_file.read()

            st.video(video_bytes)
