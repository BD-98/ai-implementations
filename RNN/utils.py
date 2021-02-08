import torch 
from unidecode import unidecode 
import glob 
import os 
import string 
from random import choice 

data_chars = string.ascii_letters + " .,:'"
path = "char-rnn-data/names/*.txt"
categ_lines = {}
all_categs = []
# Helper Functions 
findfiles = lambda path: glob.glob(path)
remove_accents = lambda char: unidecode(char)
char2index = lambda char: data_chars.find(char)
def readlines(f):
    lines = open(f, encoding="utf-8").read().strip().split()
    return lines 

for f in findfiles(path):
    categ = os.path.splitext(os.path.basename(f))[0]
    all_categs.append(categ)
    lines = readlines(f)
    categ_lines[categ] = lines 

n_categs = len(all_categs)
n_chars = len(data_chars)

# Conver to tensors  
def char2tensor(char):
    tensor = torch.zeros(1, n_chars)
    tensor[0][char2index(char)] = 1 
    return tensor 

def line2tensor(line):
    tensor = torch.zeros(len(line), 1, n_chars)
    for li, char in enumerate(line):
        tensor[li][0][char2index(char)] = 1
    return tensor

def categ_from_out(out):
    top_n, top_i = out.topk(1)
    categ_i = top_i[0].item()
    print(top_n)
    return all_categs[categ_i], categ_i

def gen_rand_training_example():
    categ = str(choice(all_categs))
    line = choice(categ_lines[categ])
    categ_tensor = torch.tensor([all_categs.index(categ)], dtype=torch.long)
    li_tensor = line2tensor(line)
    return categ, line, categ_tensor, li_tensor

