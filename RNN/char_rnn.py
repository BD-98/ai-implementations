import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from tqdm import trange 
from utils import *


# Vanilla RNN Model 

class RNN(nn.Module):
    def __init__(self, in_size, h_size, out_size):
        super(RNN, self).__init__()
        self.h_size = h_size
        self.i2h = nn.Linear(in_size + h_size, h_size)
        self.i2o = nn.Linear(in_size + h_size, out_size)
    
    def forward(self, x, h):
        combined = torch.cat((x, h), 1)
        h = self.i2h(combined)
        x = self.i2o(combined)
        x = F.log_softmax(x, dim=1)
        return x, h
    
    def init_h(self):
        return torch.zeros(1, self.h_size) 
n_hid = 128 
rnn = RNN(n_chars, n_hid, n_categs)




# Training Function 
opt = torch.optim.SGD(rnn.parameters(), lr=5e-3)
loss_function = nn.NLLLoss()
def train(categ_tensor, line_tensor):
    h = rnn.init_h()

    for i in range(line_tensor.size(0)):
        out, h = rnn(line_tensor[i], h)
    
    loss = loss_function(out, categ_tensor)

    # Update Weights 
    opt.zero_grad()
    loss.backward()
    opt.step()

    return out, loss.item()



# Training Loop 
epochs = 100000
steps = epochs / 5 
categ, line, categ_tensor, line_tensor = gen_rand_training_example()

