import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

class Network(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Conv2d(1,10,3,stride=1,padding=1),
        nn.ReLU(),
        nn.Conv2d(10,10,3,stride=1,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Conv2d(10,10,3,stride=1,padding=1),
        nn.ReLU(),
        nn.Conv2d(10,10,3,stride=1,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Flatten(),
        nn.Linear(490,10)
    )
  
  def forward(self,x):
    return self.layers(x)
  
def classification(output):
  max = output[0]
  maxi = 0
  for i in range(10):
    if output[i] > max:
      max = output[i]
      maxi = i
  return maxi

dataset = torchvision.datasets.FashionMNIST(root='./data',download=True, transform=transforms.ToTensor(),train=False)
loader = DataLoader(dataset,shuffle=True,batch_size=32)
model = Network()
model.load_state_dict(torch.load('model.pth',weights_only=True))
model.eval()
errors = 0
n = 0
for X,y in loader:
  pred = model(X)
  predicted_classes = [classification(p) for p in pred]
  for p,a in zip(predicted_classes,y):
    n += 1
    if p != a:
      errors += 1

print('Accuracy:', 1 - errors/n)