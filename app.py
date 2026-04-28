from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import base64

app = FastAPI()

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 10, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(10, 10, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(10, 10, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(10, 10, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(490, 10)
        )

    def forward(self, x):
        return self.layers(x)

model = Network()
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

class ImageInput(BaseModel):
    image_base64: str

@app.get("/")
def home():
    return {"message": "CNN FashionMNIST API working"}

@app.post("/predict")
def predict(data: ImageInput):
    image_bytes = base64.b64decode(data.image_base64)
    image = Image.open(io.BytesIO(image_bytes))

    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return {"prediction": int(predicted.item())}