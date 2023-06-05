import os
import whisper
import numpy as np

from dotenv import load_dotenv, find_dotenv
from pydub import AudioSegment
from pywhispercpp.model import Model

load_dotenv(find_dotenv())
WHISPER_MODEL = os.environ.get('WHISPER_MODEL')
WHISPER_THREADS = os.environ.get('WHISPER_THREADS')
WHISPER_LANGUAGE = os.environ.get('WHISPER_LANGUAGE')
WHISPER_DEVICE = os.environ.get('WHISPER_DEVICE')


def recognize(file):
    if WHISPER_DEVICE == "cpu":
        audio_array = audio_to_array(file.file)
        model = Model(WHISPER_MODEL, n_threads=int(WHISPER_THREADS), language=WHISPER_LANGUAGE)
        segments = model.transcribe(audio_array)
    elif WHISPER_DEVICE == "cuda":
        audio_file = save_temp_audio(file)
        model = whisper.load_model(WHISPER_MODEL, WHISPER_DEVICE)
        segments = model.transcribe(audio_file, language=WHISPER_LANGUAGE)["segments"]
        os.remove(audio_file)
    else:
        print("Incorrect WHISPER_DEVICE")
        exit(1)

    return segments


def audio_to_array(file):
    sound = AudioSegment.from_file(file)
    sound = sound.set_frame_rate(16000).set_channels(1)
    channel_sounds = sound.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds]
    arr = np.array(samples).T.astype(np.float32)
    arr /= np.iinfo(samples[0].typecode).max

    return arr


def save_temp_audio(file):
    file_location = f"temp_audio/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    return file_location
