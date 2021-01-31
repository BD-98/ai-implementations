import torch 
import torch.nn as nn
from data_utils import device 
# Dimensions 
latent_size = 64 
n_channels = 3 
image_size = 256 



class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0), # 4 X 4
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=4, padding=0), # 16 X 16 
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=4, padding=0), # 64 X 64 
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, n_channels, kernel_size=4, stride=4, padding=0), # 256 X 256 
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.main(x)
        return x 

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(n_channels,latent_size, kernel_size=5, stride=2, padding=0), # 126 X 126 
            nn.BatchNorm2d(latent_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(latent_size, 128, kernel_size=3, stride=2, padding=0), # 62 X 62 
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=0), # 29 X 29 
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=0), # 15 X 15 
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=3,stride=2, padding=0), # 6 X 6
            nn.Sigmoid()  
            )
    
    def forward(self, x):
        x = self.main(x)
        return x 

