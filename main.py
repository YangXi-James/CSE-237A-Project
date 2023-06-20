from flask import Flask, jsonify, send_file

import os
import gc
import time

import subprocess
import requests

import signal

from timeloop import Timeloop
from datetime import timedelta

import sensors.temperature as temperature
import sensors.air as air_quality
import yolov5.head_count as head_count

# Define the endpoint for the remote server
# when running locally, please change the IP to your IP in the local network.
# Run command:
#   ifconfig
# to check the ip and replace here. If you change the port defined in the
# `speech_diarization_server.py` file, please change the port as well.
REMOTE_SERVER_ENDPOINT = 'http://192.168.182.46:5000/process_audio'

# Define the duration of the audio to record in seconds
DURATION = 5

# Initialize Flask app
app = Flask(__name__)

tl = Timeloop()

# create a folder named "captured_images" in the current directory
folder_name = "examples"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
file_name = "captured_image.jpg"
file_path = os.path.join(folder_name, file_name)

# Initialize YOLO model
cap = head_count.open_camera()
head_count.capture_image(cap, file_name, view_img=False)

model = head_count.init(file_name)

# audio file name
audio_file_path = str(os.path.join(folder_name, 'audio.wav'))


# legacy codes for using camera streams directly
# model, dataset = head_count.init()
# dataset_iter = iter(dataset)


@app.route('/')
def index():
    return send_file('static/index.html')


# Endpoint to get temperature measurement
@app.route('/temperature')
def get_temperature():
    temp = temperature.read_temperature()
    if temp is not None:
        return jsonify({'temperature': temp})
    else:
        return jsonify({'error': 'Failed to retrieve temperature'})


# Endpoint to get air quality measurement
@app.route('/air_quality')
def get_air_quality():
    aq = air_quality.read_air_quality()
    if aq is not None:
        return jsonify({'air_quality': aq})
    else:
        return jsonify({'error': 'Failed to retrieve air quality'})


@app.route('/head_count')
def get_head_count():
    head_count.capture_image(cap, file_path)
    hc = head_count.read_head_count(model, file_path, view_img=False)
    if hc is not None:
        return jsonify({'head_count': hc})
    else:
        return jsonify({'error': 'Failed to retrieve air quality'})


@app.route('/speech_diarization')
def get_speech_diarization():
    # Send the audio file to the remote server for processing
    with open(audio_file_path, 'rb') as audio_file:
        response = requests.post(REMOTE_SERVER_ENDPOINT, files={'audio': audio_file})

    # Check if the processing was successful and forward the result to the client
    if response.ok:
        result = response.json()
        # Forward the result to the client using Flask or some other method
        return jsonify({'speech_diarization': result})
    else:
        print('Error processing audio')
        return jsonify({'speech_diarization': {}})


@tl.job(interval=timedelta(seconds=DURATION + 5))
def record_audio():
    # Record audio using arecord and save it to a file
    subprocess.run(['arecord', '--format=S16_LE', '-d', str(DURATION), '-t', 'wav', '-r', '16000', audio_file_path])
    time.sleep(DURATION + 2)


def cleanup():
    print("cleaning up.")
    head_count.close_camera()
    tl.stop()
    gc.collect()


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, cleanup)
    tl.start()

    app.run(debug=False, host='0.0.0.0', port=12300)
