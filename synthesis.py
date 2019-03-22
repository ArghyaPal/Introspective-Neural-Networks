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

class Synthesis():
    def __init__(self, init_std=0.3):
        self.init_std = init_std;
        
    def sample(self, module, num_iter=10, learning_rate=1e-2, add_noise=True):
        assert isinstance(module.X, torch.nn.Parameter), 'Expected X to be an instance of torch.nn.Parameter';
        
        module.train(False);
        
        # we do not want to create a graph and do backprop on net parameters, since we need only gradient of X
        for name, param in module.named_parameters():
            if name != 'X':
                param.requires_grad = False;
            else:
                param.requires_grad = True;
        
        module.X.data = module.X.data.normal_(mean=0, std=self.init_std);
        #opt = torch.optim.SGD([module.X], lr=learning_rate);
        opt = torch.optim.Adam([module.X], lr=learning_rate, amsgrad=True, betas=(0.5, 0.999));
        std_noise = learning_rate;
        
        for i in range(num_iter):
            opt.zero_grad();
            classes = -module.to_synth();
            for j in range(classes.shape[0]):
                classes[j, j, ...].backward(retain_graph=True);
                
            if add_noise:
                module.X.data += torch.empty_like(module.X.data).normal_(mean=0, std=std_noise);
                std_noise *= 0.9; 
            opt.step()
            
        module.train(True);
        
        for name, param in module.named_parameters():
            if name != 'X':
                param.requires_grad = True;
            else:
                param.requires_grad = False;
        
        return module.X.data;