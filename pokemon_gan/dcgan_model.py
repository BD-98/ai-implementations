import torch 
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

m = nn.ConvTranspose2d(3, 64, kernel_size=5, stride=1, padding=0)
rand = torch.randn(16, 3, 256, 256)
out = m(rand)
print(out.size())