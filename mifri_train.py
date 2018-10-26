import os
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import copy
import mifri_class

ap = argparse.ArgumentParser(description='mifri_train.py')

ap.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
ap.add_argument('--gpu', dest="gpu", action="store", default="gpu")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=3)
ap.add_argument('--arch', dest="arch", action="store", default="densenet121", type = str)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=512)

mf = ap.parse_args()
where = mf.data_dir
path = mf.save_dir
learning_rate = mf.learning_rate
arch = mf.arch
dropout = mf.dropout
hidden_units = mf.hidden_units
power = mf.gpu
epochs = mf.epochs

train_loader, valid_loader, test_loader = mifri_class.load_data()

model, optimizer = mifri_class.network_setup(arch,dropout,hidden_units,learning_rate)

mifri_class.training_the_network(model, optimizer, epochs, train_loader)

mifri_class.saving_the_checkpoint(path,structure,hidden_units,dropout,learning_rate)

print('***********************************')
print("Done. You have a trained model now!")
print('***********************************')