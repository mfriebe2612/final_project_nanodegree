# ***********************************
# IMAGE CLASSIFIER UDACITY NANODEGREE
# Michael Friebe
# IMPORTS HERE
# ***********************************
#

import os

import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torchvision import datasets, transforms, models

import json

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

import argparse
import time
import copy


print("**************")
print('IMPORTS Loaded')
print("**************")


def load_data():
    # ********************************************************
    # IMAGE / TRAINING / VALIDATION / TESTING file directories
    # ********************************************************
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # *********************************************************************
    # Define your transforms for the training, validation, and testing sets
    # transforms for the train dataset
    # *********************************************************************
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])
                                        ])

    valid_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])
                                        ])

    test_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                          [0.229, 0.224, 0.225])])
    
    # ************************************    
    # LOADING THE DATASET WITH IMAGEFOLDER
    # ************************************

    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transform)
    test_data  = datasets.ImageFolder(test_dir, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    test_loader  = torch.utils.data.DataLoader(test_data, batch_size=32)


    # *************************************************************+************
    # How many images are in the dataset / what is the training and testing rate
    # **************************************************************************
    train_dataset_images = len(train_data.imgs)
    valid_dataset_images = len(valid_data.imgs)
    test_dataset_images = len(test_data.imgs)
    print()
    print('---------------------------------------------------------------')
    print("Number of images in the ENTIRE dataset: ", train_dataset_images + valid_dataset_images + test_dataset_images)
    print("Number of images in the TRAINING dataset: ", train_dataset_images)
    print("Percentage of TRAINING data ", int(100*(train_dataset_images/( train_dataset_images + valid_dataset_images + test_dataset_images)))," %")
    print("Number images in VALIDATION dataset:", valid_dataset_images)
    print("Percentage of VALIDATION data ", int(100*(valid_dataset_images/( train_dataset_images + valid_dataset_images + test_dataset_images)))," %")
    print("Number images in the TESTING dataset:", test_dataset_images)
    print("Percentage of TESTING data ", int(100*(test_dataset_images/( train_dataset_images + valid_dataset_images + test_dataset_images))), " %")
    print('---------------------------------------------------------------')
    print()


    # *******************************
    # load the JSON categories file
    # *******************************
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    print()
    print(cat_to_name)
    print()

    return(train_loader, valid_loader, test_loader)


def network_setup(arch,dropout,hidden_units,learning_rate):
    # ****************************
    # Build and train your network
    # Load a pre-trained network 
    # ****************************

    arch = 'densenet121'

    model = getattr(models, arch)(pretrained=True)
    input_nodes = {'densenet121':1024, 'vgg13': 25088, 'vgg16': 25088, 'alexnet': 9216}
    print(model)

    # *************************************************************+
    # Parameters
    hidden_units = 512

    # Define a new, untrained feed-forward network as a classifier, 
    # using ReLU activations and dropout and not updated parameters
    for param in model.parameters():
        param.requires_grad = False

    # RELU used as Activationfunction between the layers
    # Softmax Function for the OUTPUT layer
    # OUTPUT = 102 (=number of image files)
    # for other architectures additional hiddenlayers should be integrated
    classifier = nn.Sequential(OrderedDict([('first_node', nn.Linear(input_nodes[arch], hidden_units)),
                                            ('relu', nn.ReLU()),
                                            ('dropout',nn.Dropout(p=0.5)),
                                            ('hidden_layer', nn.Linear(hidden_units, 102)),
                                            ('relu', nn.ReLU()),
                                            ('output_function', nn.LogSoftmax(dim=1))
                            ]))
    
    model.classifier = classifier



    # ******************************************************************************
    # opimzer and criterion functions
    # ******************************************************************************

    # Loss and optimizer functions
    criterion = nn.NLLLoss()

    # ADAM or SGD (optim.SGD())
    learning_rate = 0.001 # best around 0.001 - 0.005 -- with 3 - 5 EPOCHS will generate > 70% accuracy
    lr = learning_rate
    optimizer = optim.Adam(model.classifier.parameters(), lr)


    # **********
    # GPU or CPU
    # **********

    cuda = torch.cuda.is_available()
    if cuda: # if TRUE THEN GPU
        model.cuda()
        print('************************************************************************')
        print("The NETWORK runs on graphic processor. The total number of GPU's:", (torch.cuda.device_count()))
        print('************************************************************************')
    else: # FALSE, THEN CPU
        model.cpu()
        print('************************************************************')
        print("Unfortunately no Graphic Processor available -- CPU is used!")
        print('************************************************************')

    return(criterion, optimizer)



# *****************************
# NETWORK TRAINING + PARAMETERS
# *****************************

def training_the_network(model, optimizer, epochs, train_loader):
    epochs = 5 # for the densenet network architecture 3 is the minimum to reach 70% accuracy
    epoch_finish = epochs # reserve variable for a future break of the loop, if a ertain percentage of accuracy is reached
    desired_accuracy = 0.7 # the desired accuracy in percent / 100
    steps = 0 # count the number of steps
    training_loss = 0 # set the initial training_loss
    print_every = 50 # show results every XX times

    since = time.time() # Start the clock
    cuda = torch.cuda
    
    for e in range(epochs): 
    
        # Model in training mode
        model.train()
    
        for mf1, (inputs, labels) in enumerate(train_loader):
            steps += 1
                
    
            inputs = Variable(inputs)
            labels = Variable(labels)
        
            # resetting the gradient -- otherwise the optimizer tensor will accumulate over the epochs
            optimizer.zero_grad()
            
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
        
            # Training -- FORWARD PASS
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        
            training_loss += loss.data[0]  
        
            # Show results for print_every number of steps
            if steps % print_every == 0:
                model.eval()
                accuracy = 0            
                valid_loss = 0
            
                # VALIDATION Loop
                for mf2, (inputs, labels) in enumerate(valid_loader):
                
                    inputs = Variable(inputs, requires_grad=False, volatile = True)
                    labels = Variable(labels)
        
                    if cuda:
                        # GPU available
                        inputs, labels = inputs.cuda(), labels.cuda()
            
                    outputs = model.forward(inputs)
                    loss = criterion(outputs, labels)
                
                    valid_loss += loss.data[0]
                
                    ps = torch.exp(outputs).data
                
                    equality = (labels.data == ps.max(1)[1])
               
                    accuracy += equality.type_as(torch.FloatTensor()).mean()
            
          
                print()
                print('****************************************************')
                print("' Epoch: {}/{}   ".format(e+1, epochs),
                    "'\n Training Loss: {:.3f}   ".format(training_loss),
                    "'\n Validation Loss: {:.3f}   ".format(valid_loss),
                    "'\n Valid Accuracy %: {:.3f}   ".format(100*accuracy/len(valid_loader)),
                    "'\n All in all {} steps up to now!".format(int(steps)))
                print('****************************************************')
            
                # Reset training_loss
                training_loss = 0
            
                # Back to training mode
                model.train()
 
    time_elapsed = time.time() - since # Stop the clock

    print('*****************************************')
    print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
    print('*****************************************')

    return


# *******************************************
# SAVE the NETWORK in the file checkpoint.pth
# *******************************************
def saving_the_checkpoint(path,structure,hidden_units,dropout,learning_rate):
    network_name = 'checkpoint.pth'
    filename = network_name

    checkpoint = {'model': model.state_dict(), 
                'optimizer': optimizer.state_dict(), 
                'class_to_idx': train_data.class_to_idx,
                'arch': arch,
                'hidden_units': hidden_units,
                'learning_rate': learning_rate
                }

    torch.save(checkpoint, network_name)


    print("****************")
    print('Checkpoint Saved')
    print("****************") 
    print(network_name)

    return

# *********************************************************************
# TODO: Write a function that loads a checkpoint and rebuilds the model
# *********************************************************************
# below the function DEF

def loading_the_checkpoint(path='checkpoint.pth'):

    checkpoint = torch.load(filename)    
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.load_state_dict(checkpoint['model'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model, optimizer




# ***************************
# IMAGE PRE-PROCESSING
# ***************************
# Function call process_image
# ***************************

def process_image(image):
    
    # Open image file
    mf = Image.open(image)
    # Resize the images where the shortest side is 256 pixels
    mf = mf.resize((256,256))
    # Crop out the center 224x224 portion of the image 
    value = 0.5*(256-224)
    mf = mf.crop((value,value,256-value,256-value))
    # Normalize:  0-255, but the model expected floats 0-1
    # Convert image to an array and divide each element
    mf = np.array(mf)/255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    mf = (mf - mean) / std
    
    return mf.transpose(2,0,1)



# ************************
# IMAGE Show Function Call
# ************************

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    image = image.transpose((1, 2, 0))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


# ***********************************************************************************
# PREDICTION Function Call -- predict the category of an image with the learned model
# ***********************************************************************************

# topk = X, where X is the number for the X most often used categories / classes

def predict(image_path, model, topk=5):
    
    cuda = torch.cuda.is_available()
    if cuda:
        model.cuda()
        print()
        print('*********')
        print("GPU used!")
        print('*********')
        print()
    else:
        model.cpu()
        print()
        print('*********')
        print("GPU used!")
        print('*********')
        print()

   
    
    # turn off dropout
    model.eval()

    # The image
    image = process_image(image_path)
    
    # tranfer to tensor
    image = torch.from_numpy(np.array([image])).float()
    
    # The image becomes the input
    image = Variable(image)
    if cuda:
        image = image.cuda()
        
    output = model.forward(image)
    
    probabilities = torch.exp(output).data
    
    prob = torch.topk(probabilities, topk)[0].tolist()[0]
    index = torch.topk(probabilities, topk)[1].tolist()[0]
    
    ind = []
    for mf4 in range(len(model.class_to_idx.items())):
        ind.append(list(model.class_to_idx.items())[mf4][0])

    # transfer index to label
    label = []
    for mf5 in range(5):
        label.append(ind[index[mf5]])

    return prob, label