# ref: https://huggingface.co/spaces/minhdang/test/blob/main/app.py

import fastapi
import pydantic
import requests


class ImagePrompt(pydantic.BaseModel):
    query: str
    imageBase64: str


app = fastapi.FastAPI()


@app.post("/image_chat")
def imageCaption(image_prompt: ImagePrompt):
    data = dict(
        model="moondream",
        prompt=image_prompt.query,
        stream=False,
        images=[image_prompt.imageBase64],
    )
    response = requests.post("http://localhost:11434/api/generate", data=data)  # ollama
    return dict(response = response.json()['response'])
