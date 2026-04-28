from fastapi import FastAPI
from pydantic import BaseModel
import requests
import base64
import io
from PIL import Image
import torchvision.transforms as transforms

app = FastAPI()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

class ImageInput(BaseModel):
    image_base64: str

@app.post("/predict")
def predict(data: ImageInput):
    image_bytes = base64.b64decode(data.image_base64)
    image = Image.open(io.BytesIO(image_bytes))

    tensor = transform(image).unsqueeze(0).tolist()

    response = requests.post(
        "http://inference:8000/predict",
        json={"tensor": tensor}
    )

    return response.json()