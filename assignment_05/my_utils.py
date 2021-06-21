import os
import random
import torch
import numpy as np

# Setting the computing device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_path = './'

def set_random_seed(random_seed=None):
    """Using random seed for numpy and torch
    """
    if(random_seed is None):
        random_seed = 13
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    return

def load_model(path):
    path = basepath+path
    sm = torch.load(path)
    return sm['model'], sm['train_loss'], sm['val_loss'], sm['loss_iters'], \
           sm['valid_acc']