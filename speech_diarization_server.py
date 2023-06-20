from flask import Flask, request, jsonify
import os
import time
import sensors.nemo_speech_diarization as sd

app = Flask(__name__)

auxiliary_file_dir = "examples/"
output_dir = os.path.join(auxiliary_file_dir, 'outputs/')
audio_path = os.path.join(auxiliary_file_dir, "test_16k.wav")

sd_model = sd.init_sd_model(auxiliary_file_dir)


@app.route('/')
def index():
    return "<h1>Speech Diarization Server is running.</h1>"


@app.route('/process_audio', methods=['POST'])
def process_audio():
    print("received info")
    # Get the audio data from the request
    audio_data = request.files['audio']

    # Save the audio file as test_16k.wav, a specified name in model config
    filename = os.path.join("./examples", "test_16k.wav")
    # Write the audio data to a file
    # with open(filename, 'wb') as f:
    #     f.write(audio_data)
    audio_data.save(filename)
    time.sleep(5)

    try:
        # Call the diarization function with the filename and get the result
        result = sd.diarization(sd_model, output_dir)
        print(result)

        # Return the result as a JSON response
        return result
    except ValueError as e:
        print(e)
        return {}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
