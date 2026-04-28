from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn

app = FastAPI()

# SAME model as train.py
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

class TensorInput(BaseModel):
    tensor: list

@app.post("/predict")
def predict(data: TensorInput):
    x = torch.tensor(data.tensor).float()

    with torch.no_grad():
        outputs = model(x)
        _, predicted = torch.max(outputs, 1)

    return {"prediction": int(predicted.item())}