import os

from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, UploadFile
from speech_recognition import recognize

load_dotenv(find_dotenv())
WHISPER_DEVICE = os.environ.get('WHISPER_DEVICE')

app = FastAPI()


@app.post("/speech2text")
async def speech2text(file: UploadFile):
    segments = recognize(file)
    if WHISPER_DEVICE == "cpu":
        text = list(map(lambda x: x.text, segments))
    elif WHISPER_DEVICE == "cuda":
        text = list(map(lambda x: x["text"], segments))
    else:
        exit(1)

    return text
