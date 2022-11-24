import torch
from torch.utils import data
import pandas as pd

# We read the dataset and create an iterable.
class MyPoints(data.Dataset):
    def __init__(self, filename):
        pd_data = pd.read_csv(filename).values   # Read data file.
        self.data = pd_data[:,:2]   # 1st and 2nd columns --> x,y
        self.target = pd_data[:,2:]  # 3rd column --> label
        self.n_samples = self.data.shape[0]
    
    def __len__(self):   # Length of the dataset.
        return self.n_samples
    
    def __getitem__(self, index):   # Function that returns one point and one label.
        return torch.Tensor(self.data[index]), torch.Tensor(self.target[index])