import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from utils import

# Model 
class RNN(nn.Module):
    def __init__(self, in_size, h_size, out_size):
        super(RNN, self).__init__()
        self.h_size = h_size 
        self.i2h = nn.Linear(in_size + h_size, h_size)
        self.i2o = nn.Linear(in_size + h_size, out_size)
    
    def forward(self, x, h):
        comb = torch.cat((x, h), 1)
        h = self.i2h(comb)
        out = self.i2o(comb)
        out = F.log_softmax(out, dim=1)
        return out, h 
    
    def init_h(self):
        return torch.zeros(1, self.h_size)
    
#Hyp
# Training Function 
