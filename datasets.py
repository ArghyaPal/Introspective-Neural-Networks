import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST

class MnistDataset(Dataset):
    def __init__(self, root_dir='data/', training='train', train_samples='all'):
        self.imgs = [];
        self.labels = [];
        self.fakes = [];
        
        if training == 'train' or training == 'validate':
            x, y = torch.load(root_dir + 'processed/training.pt');
        else:
            x, y = torch.load(root_dir + 'processed/test.pt');
        if train_samples == 'all' and training == 'train':
            for i in range(50000):
                self.imgs.append(x[i, ...].float()/255);
                self.labels.append(y[i].long());
                self.fakes.append(torch.ByteTensor([0])[0])
        elif training == 'train':
            for i in range(train_samples):
                self.imgs.append(x[i, ...].float()/255);
                self.labels.append(y[i].long());
                self.fakes.append(torch.ByteTensor([0])[0]);
        elif training == 'validate':
            for i in range(1, 10001):
                self.imgs.append(x[-i, ...].float()/255);
                self.labels.append(y[-i].long());
                self.fakes.append(torch.ByteTensor([0])[0]);
        elif training == 'test':
            for i in range(y.shape[0]):
                self.imgs.append(x[i, ...].float()/255);
                self.labels.append(y[i].long());
                self.fakes.append(torch.ByteTensor([0])[0]);
                
    def __len__(self):
        return len(self.labels);

    def __getitem__(self, idx):
        x = self.imgs[idx];
        y = self.labels[idx];
        fake = self.fakes[idx];
        
        return (x.unsqueeze(0)-0.5)/(0.5)*0.6, y.float().unsqueeze(0), fake;
    
    def add_artificial(self, X):
        for i in range(X.shape[0]):
            self.imgs.append(X[i, 0, ...].detach().cpu());
            self.labels.append(self.labels[0].new(1).fill_(-i)[0]);
            self.fakes.append(torch.ByteTensor([1])[0]);
        pass;
    
def get_data_loaders(train_batch_size, val_batch_size, test_batch_size, train_size='all'):
    MNIST(download=True, train=True, root=".").train_data.float();

    train_loader = DataLoader(MnistDataset(root_dir='', training='train', train_samples=train_size),
                              batch_size=train_batch_size, shuffle=True);

    val_loader = DataLoader(MnistDataset(root_dir='', training='validate'),
                            batch_size=val_batch_size, shuffle=False);
    
    test_loader = DataLoader(MnistDataset(root_dir='', training='test'),
                            batch_size=test_batch_size, shuffle=False);
    return train_loader, val_loader, test_loader;
