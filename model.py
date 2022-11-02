import torch.nn
import torch.nn as nn

encoder = torch.nn.Sequential(
    nn.Conv2d(1,64,3),
    nn.ReLU(),
    nn.Conv2d(64,64,3, stride=2),
    nn.ReLU(),
    nn.Conv2d(64,64,3),
    nn.MaxPool2d(6)
)

projector = torch.nn.Sequential(
    nn.Linear(64,128),
    nn.ReLU(),
    nn.Linear(128,256),
    nn.ReLU(),
    nn.Linear(256,512)
)

test_head = nn.Linear(64,10)