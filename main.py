from fastapi import FastAPI, UploadFile

from speech_recognition import recognize

app = FastAPI()


@app.post("/speech2text")
async def speech2text(file: UploadFile):
    segments = recognize(file)
    text = list(map(lambda x: x.text, segments))

    return text
