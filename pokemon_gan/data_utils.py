import torch 
import torchvision
from torchvision import transforms
# Device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Data
transform = transforms.ToTensor()
root = "~/coding/learning/ai-implementations/datasets/pokemon"
trainset = torchvision.datasets.ImageFolder(root=root, transform=transform)
# 819 256 X 256 Images
data = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=16) 
