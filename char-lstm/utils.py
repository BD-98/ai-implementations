import torch
from unidecode import unidecode 
import os 
import string 
from glob import glob
import re  
# Data
path = "../datasets/char-rnn-data/names/*.txt"
letters = string.ascii_letters + " .,:"

# Categories 
categ_li = {}
all_categs = []

# Helper Functions 
findfiles = lambda f: glob(path)
def readlines(f):
    lines = open(f, encoding='utf-8').read().split()
    return lines


for f in findfiles(path):
    categ = os.path.splitext(os.path.basename(f))[0]