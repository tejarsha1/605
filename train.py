import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import sys

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
  
dataset = torchvision.datasets.FashionMNIST(root='./data',download=True, transform=transforms.ToTensor())
loader = DataLoader(dataset,shuffle=True,batch_size=32)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Network()
model.to(device)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),.1)

if len(sys.argv) > 1:
  epochs = int(sys.argv[1])
else: 
  epochs = 10
model.train()
for epoch in range(epochs):
  for batch,(X,y) in enumerate(loader):
    print(epoch,batch)
    pred = model(X)
    l = loss(pred,y)
    l.backward()
    optimizer.step()
    optimizer.zero_grad()

torch.save(model.state_dict(),'model.pth')