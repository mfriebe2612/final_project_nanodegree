import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim, tensor
import torch.nn.functional as Funct
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as model
from collections import OrderedDict
import json
import PIL
from PIL import Image
import mifri_class
import argparse

ap = argparse.ArgumentParser(description='predict-file')
ap.add_argument('input_img', default='', nargs='*', action="store", type = str)
ap.add_argument('checkpoint', default='', nargs='*', action="store",type = str)
ap.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
ap.add_argument('--category_names', dest="category_names", action="store", default='name_cat.json')
ap.add_argument('--gpu', default="gpu", action="store", dest="gpu")

mf = ap.parse_args()
path_image = mf.input_img
number_of_outputs = mf.top_k
power = mf.gpu
input_image = mf.input_img
path = mf.checkpoint

training_loader, testing_loader, validation_loader = mifri_class.load_data()

mifri_class.loading_the_checkpoint(path)

with open('name_cat.json', 'r') as json_file:
    name_cat = json.load(json_file)

probabilities = mifri_class.predict(path_image, model, number_of_outputs, power)

labels = [name_cat[str(index + 1)] for index in np.array(probabilities[1][0])]
probability = np.array(probabilities[0][0])

i=0
while i < number_of_outputs:
    print("{} with a probability of {}".format(labels[i], probability[i]))
    i += 1

print('*****')
print("Done!")
print('*****')