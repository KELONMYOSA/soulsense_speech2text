import os

import numpy as np
from dotenv import load_dotenv, find_dotenv
from pydub import AudioSegment
from pywhispercpp.model import Model

load_dotenv(find_dotenv())
WHISPER_MODEL = os.environ.get('WHISPER_MODEL')
WHISPER_THREADS = os.environ.get('WHISPER_THREADS')


def recognize(file):
    audio_array = audio_to_array(file.file)

    model = Model(WHISPER_MODEL, n_threads=int(WHISPER_THREADS), language="ru")
    segments = model.transcribe(audio_array)

    return segments


def audio_to_array(file):
    sound = AudioSegment.from_file(file)
    sound = sound.set_frame_rate(16000).set_channels(1)
    channel_sounds = sound.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds]
    arr = np.array(samples).T.astype(np.float32)
    arr /= np.iinfo(samples[0].typecode).max
    return arr
