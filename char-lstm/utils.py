import torch
from unidecode import unidecode 
import os 
import string 

# Data
path = "../datasets/char-rnn-data/names/*.txt"
letters = string.ascii_letters + " .,:"

# Helper Functions 
remove_accents = lambda s: unidecode(s) 
findfiles = lambda files: glob(files)
print(letters)
