import os
import wget
import json

from nemo.collections.asr.models.msdd_models import NeuralDiarizer

from omegaconf import OmegaConf

import librosa
import soundfile as sf


def init_sd_model(auxiliary_file_dir):
    data_dir = os.path.join(auxiliary_file_dir, "data/")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    init_input_manifest(auxiliary_file_dir)

    pretrained_vad = 'vad_multilingual_marblenet'
    pretrained_speaker_model = 'titanet_large'

    MODEL_CONFIG = os.path.join(data_dir, 'diar_infer_telephonic.yaml')
    if not os.path.exists(MODEL_CONFIG):
        config_url = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_telephonic.yaml"
        MODEL_CONFIG = wget.download(config_url, data_dir)

    config = OmegaConf.load(MODEL_CONFIG)

    config.num_workers = 1  # Workaround for multiprocessing hanging with ipython issue

    output_dir = os.path.join(auxiliary_file_dir, 'outputs/')
    config.diarizer.manifest_filepath = os.path.join(auxiliary_file_dir, "input_manifest.json")
    config.diarizer.out_dir = output_dir  # Directory to store intermediate files and prediction outputs

    config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
    config.diarizer.oracle_vad = False  # compute VAD provided with model_path to vad config
    config.diarizer.clustering.parameters.oracle_num_speakers = False

    # Here, we use our in-house pretrained NeMo VAD model
    config.diarizer.vad.model_path = pretrained_vad
    config.diarizer.vad.parameters.onset = 0.8
    config.diarizer.vad.parameters.offset = 0.6
    config.diarizer.vad.parameters.pad_offset = -0.05

    config.diarizer.msdd_model.model_path = 'diar_msdd_telephonic'  # Telephonic speaker diarization model
    config.diarizer.msdd_model.parameters.sigmoid_threshold = [0.7, 1.0]  # Evaluate with T=0.7 and T=1.0

    print(OmegaConf.to_yaml(config))

    sd_model = NeuralDiarizer(cfg=config)
    return sd_model


def init_input_manifest(auxiliary_file_dir):
    input_manifest_path = os.path.join(auxiliary_file_dir, "input_manifest.json")
    if not os.path.exists(input_manifest_path):
        meta = {
            'audio_filepath': os.path.join(auxiliary_file_dir, "test_16k.wav"),
            'offset': 0,
            'duration': None,
            'label': 'infer',
            'text': '-',
            'num_speakers': None,
            'rttm_filepath': None,
            'uem_filepath': None
        }

        with open(input_manifest_path, "w") as fp:
            json.dump(meta, fp)
            fp.write("\n")

        print(f"input manifest json file is created ar {input_manifest_path}")


def diarization(sd_model, output_dir):
    sd_model.diarize()
    rttmf = os.path.join(output_dir, "pred_rttms/test_16k.rttm")
    return rttm_to_json(rttmf)


def rttm_to_json(rttm_file):
    speakers = {}

    # Parse the RTTM file
    with open(rttm_file, 'r') as f:
        for line in f:
            fields = line.strip().split()
            if fields[0] == 'SPEAKER':
                # Add speaker information to the JSON object
                _, _, _, start_time, duration, _, _, speaker_id, _, _ = fields
                if speaker_id in speakers:
                    speakers[speaker_id] += float(duration)
                else:
                    speakers[speaker_id] = float(duration)

    # # Save the JSON object to a file
    # with open('output.json', 'w') as f:
    #     json.dump(speakers, f)
    return speakers


def downsample(input_audio_file, output_audio_file, orig_sr, target_sr):
    # load the audio file at 48 kHz sampling rate
    y, sr = librosa.load(input_audio_file, sr=orig_sr)

    # resample the audio to 16 kHz sampling rate
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    # save the resampled audio to a file
    sf.write(output_audio_file, y_resampled, target_sr)


if __name__ == "__main__":
    auxiliary_file_dir = "../examples/"
    output_dir = os.path.join(auxiliary_file_dir, 'outputs/')
    audio_path = os.path.join(auxiliary_file_dir, "test_16k.wav")

    if not os.path.exists(audio_path):
        downsample(os.path.join(auxiliary_file_dir, "test.wav"), os.path.join(auxiliary_file_dir, "test_16k.wav"),
                   orig_sr=48000, target_sr=16000)

    sd_model = init_sd_model(auxiliary_file_dir)
    result = diarization(sd_model, output_dir)
    print(result)
    # print(rttm_to_json("../examples/outputs/pred_rttms/test_16k.rttm "))
