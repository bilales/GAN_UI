# data_loader.py
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader as TorchDataLoader

class DataLoader:
    def __init__(self, data_folder, batch_size=32, image_size=28, download=True):
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.image_size = image_size
        self.download = download
        self.num_workers = 2 # You can adjust this based on your system

    def get_data_loader(self, train=True):
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(), # Scales data to [0,1]
            transforms.Normalize((0.5,), (0.5,))  # Normalizes to [-1,1] as MNIST is grayscale
        ])
        
        # Ensure data_folder is properly structured for MNIST
        # MNIST dataset class expects root directory where MNIST/processed and MNIST/raw will live
        dataset = MNIST(root=self.data_folder, train=train, transform=transform, download=self.download)
        
        loader = TorchDataLoader(dataset, 
                                 batch_size=self.batch_size, 
                                 shuffle=True, 
                                 num_workers=self.num_workers,
                                 pin_memory=torch.cuda.is_available()) # pin_memory for faster CPU to GPU transfer
        return loader