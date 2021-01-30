import torch 
import torchvision
from torchvision import transforms
transform = transforms.ToTensor()
root = "~/coding/learning/ai-implementations/datasets/pokemon"
trainset = torchvision.datasets.ImageFolder(root=root, transform=transform)
'''
819 256 X 256 Images
'''
pokemon_images = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=16)
