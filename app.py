import gradio as gr
from openvino.inference_engine import IECore
import cv2
import numpy as np
import os

# Load the OpenVINO model for accident detection
model_xml = 'accident_detection.xml'
model_bin = 'accident_detection.bin'

ie = IECore()
net = ie.read_network(model=model_xml, weights=model_bin)
exec_net = ie.load_network(network=net, device_name="CPU")

def detect_accident(input_video_path):
    cap = cv2.VideoCapture(input_video_path)
    
    # Initialize a counter for accident detections
    accident_frames = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        image = cv2.resize(frame, (224, 224))
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = image.reshape(1, 3, 224, 224)

        # Run inference
        outputs = exec_net.infer(inputs={'input': image})

        # Assuming the output is a binary classification
        # where 0 indicates an accident and 1 indicates no accident
        if np.argmax(outputs['output']) == 0:
            accident_frames += 1

    cap.release()

    # Setting a threshold of 10% of the video's frames to detect an accident
    # This means if more than 10% of the frames indicate an accident, 
    # the entire video will be classified as having an accident.
    if accident_frames / total_frames > 0.10:  
        return "Accident Detected!"
    else:
        return "No Accident Detected."
    
    
# Create the Gradio interface
inputs = gr.Video(label="Input Video")
outputs = gr.outputs.Textbox(label="Detection Result")

title = "Accident Detection App"
description = "Upload a video and see if an accident was detected."
#iface = gr.Interface(fn=detect_accident, inputs=inputs, outputs=outputs, title=title, description=description)
iface = gr.Interface(detect_accident, 
                    inputs=inputs, outputs=outputs, title=title, description=description,
                    cache_examples=True)


# Launch the Gradio interface
iface.launch()