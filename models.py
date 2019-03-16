import torch
import matplotlib.pyplot as plt
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.datasets import MNIST
from tqdm.autonotebook import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import inspect
import time
from torch import nn, optim
import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader
from torch.cuda import set_device
from tqdm import tnrange
from collections import OrderedDict

class MnistResNet(torch.nn.Module):
    def __init__(self):
        super(MnistResNet, self).__init__()
        
        self.net = nn.Sequential(OrderedDict([
            ('conv1', torch.nn.Conv2d(1, 64, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1), bias=False)),
            ('bn1', torch.nn.BatchNorm2d(64)),
            ('relu1', torch.nn.ReLU()),
            
            ('conv2', torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), bias=False)),
            ('bn2', torch.nn.BatchNorm2d(128)),
            ('relu2', torch.nn.ReLU()),
            
            ('conv3', torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), bias=False)),
            ('bn3', torch.nn.BatchNorm2d(256)),
            ('relu3', torch.nn.ReLU()),
            
            ('conv4', torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
            ('bn4', torch.nn.BatchNorm2d(512)),
            ('relu4', torch.nn.ReLU()),
            
            ('conv5', torch.nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)),
            ('bn5', torch.nn.BatchNorm2d(256)),
            ('relu5', torch.nn.ReLU()),
            
            ('conv6', torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
            ('bn6', torch.nn.BatchNorm2d(512)),
            ('relu6', torch.nn.ReLU()),
            
            ('conv7', torch.nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
            ('bn7', torch.nn.BatchNorm2d(256)),
            ('relu7', torch.nn.ReLU()),
            
            ('conv8', torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)),
            ('bn8', torch.nn.BatchNorm2d(512)),
            ('relu8', torch.nn.ReLU()),
        
            ('conv9', torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), bias=False)),
            ('bn9', torch.nn.BatchNorm2d(512)),
            ('relu9', torch.nn.ReLU())]));
        
        self.fc = torch.nn.Linear(512, 10);
        
        self.X = torch.nn.Parameter(torch.empty((10, 1, 28, 28)).normal_(mean=0, std=0.3));
        self.X.requires_grad = False;
        
    def forward(self, x):
        #return torch.sigmoid(self.fc(self.net(x).squeeze(-1).squeeze(-1)));
        return self.fc(self.net(x).squeeze(-1).squeeze(-1));
    
    def to_synth(self):
        return self.fc(self.net(self.X).squeeze(-1).squeeze(-1));
    
