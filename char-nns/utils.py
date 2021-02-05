import torch
from unidecode import unidecode 
import os 
import string 
from glob import glob
import re  
# Data
path = "../datasets/char-rnn-data/names/*.txt"
data_chars = string.ascii_letters + " .,:"
n_chars = len(data_chars)
# Categories 
categ_li = {}
all_categs = []

# Helper Functions 
findfiles = lambda f: glob(path)
char2index = lambda char: data_chars.find(char) 
readlines = lambda f: open(f, encoding="utf-8").read().strip().split()

for f in findfiles(path):
    categ = os.path.splitext(os.path.basename(f))[0]
    all_categs.append(categ)
    lines = readlines(f)
    categ_li[categ] = lines

def char2tensor(char):
    tensor = torch.zeros(1, n_chars)
    tensor[0][char2index(char)] = 1 
    return tensor 

def line2tensor(line):
    tensor = torch.zeros(len(line), 1, n_chars)
    for li, char in enumerate(line):
        tensor[li][0][char2index(char)] = 1 
    return tensor 

n_categs = len(all_categs)
test_t = torch.zeros(1, n_chars, 5)
