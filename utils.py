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


class MnistDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir='data/', training=True):
        self.imgs = [];
        self.labels = [];
        self.fake = [];
        
        if training:
            x, y = torch.load(root_dir + 'processed/training.pt');
        else:
            x, y = torch.load(root_dir + 'processed/test.pt');
        for i in range(y.shape[0]):
            self.imgs.append(x[i, ...].float()/255);
            self.labels.append(y[i].long());
            self.fake.append(torch.ByteTensor([0])[0]);
    
    def __len__(self):
        return len(self.labels);

    def __getitem__(self, idx):
        
        x = self.imgs[idx];
        y = self.labels[idx];
        fake = self.fake[idx];
        
        return x.unsqueeze(0), y.float().unsqueeze(0), fake;
    
    def add_artificial(self, X):
        for i in range(X.shape[0]):
            self.imgs.append(X[i, 0, ...].detach().cpu());
            self.labels.append(self.labels[0].new(1).fill_(-i)[0]);
            self.fake.append(torch.ByteTensor([1])[0])
        pass;


def get_data_loaders(train_batch_size, val_batch_size):
    mnist = MNIST(download=False, train=True, root=".").train_data.float()
    
    data_transform = Compose([ Resize((224, 224)),ToTensor(), Normalize((mnist.mean()/255,), (mnist.std()/255,))])

    train_loader = DataLoader(MnistDataset(root_dir=''),
                              batch_size=train_batch_size, shuffle=True)

    val_loader = DataLoader(MnistDataset(root_dir='', training=False),
                            batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader


    
def calculate_metric(metric_fn, true_y, pred_y):
    if "average" in inspect.getfullargspec(metric_fn).args:
        return metric_fn(true_y, pred_y, average="macro")
    else:
        return metric_fn(true_y, pred_y)
    
    
    
def print_scores(p, r, f1, a, batch_size):
    for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
        print(f"\t{name.rjust(14, ' ')}: {sum(scores)/batch_size:.4f}")    