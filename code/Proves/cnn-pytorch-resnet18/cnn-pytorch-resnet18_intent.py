from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt
import time
import os
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)




# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



data_dir = "handwriting-recognition/"

# ResNet input size
input_size = (224,224)

# Just normalization
data_transforms = {
    'train_v2': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validation_v2': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")


# Batch size for training (change depending on how much memory you have)
batch_size = 8

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train_v2', 'validation_v2']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train_v2', 'validation_v2']}


import matplotlib.image as mpimg

# show some images
plt.figure(figsize=(16, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    idx = np.random.randint(0,len(image_datasets['train_v2'].samples))
    image = mpimg.imread(image_datasets['train_v2'].samples[idx][0])
    plt.imshow(image)
    plt.axis('off')




def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    acc_history = {"train": [], "val": []}
    losses = {"train": [], "val": []}

    # we will keep a copy of the best weights so far according to validation accuracy
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train_v2', 'validation_v2']:
            if phase == 'train_v2':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    losses[phase].append(loss)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            acc_history[phase].append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, acc_history, losses





def initialize_model(num_classes):
    # Resnet18
    model = models.resnet18()

    model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True) # YOUR CODE HERE!

    input_size = 224

    return model, input_size


# Number of classes in the dataset
num_classes = 29

# Initialize the model
model, input_size = initialize_model(num_classes)

# Print the model we just instantiated
print(model)


# Send the model to GPU
model = model.to(device)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Number of epochs to train for
num_epochs = 15

optimizer_ft = optim.Adam(model.parameters(), lr=0.001)

# Train and evaluate
model, hist, losses = train_model(model, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)





"""
import os
import cv2
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.optim import Adam
from torch.nn import Linear
from torch.utils.data import DataLoader, ImageFolder
from sklearn.preprocessing import LabelEncoder

# Funció per convertir imatges a escala de grisos i redimensionar-les
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (224, 224))
    return image

# Funció per inicialitzar el model ResNet18 i modificar la capa final
def initialize_model(num_classes):
    model = resnet18(pretrained=False)
    model.fc = Linear(model.fc.in_features, num_classes)
    return model

# Cargar i transformar les dades
def create_dataloaders(data_dir, transform=None):
    image_datasets = {x: ImageFolder(os.path.join(data_dir, x), transform=transform) for x in ['train', 'val']}
    dataloaders_dict = {x: DataLoader(image_datasets[x], batch_size=8, shuffle=True, num_workers=4) for x in ['train', 'val']}
    return dataloaders_dict

# Transformacions per a les dades
train_transforms = transforms.Compose([
    transforms.Lambda(lambda img: preprocess_image(img)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Carregar les dades
train_img_dir = '/handwriting-recognition/train_v2/train'
test_img_dir = '/handwriting-recognition/validation_v2/validation'

# Crear conjunts de dades i cargadors
dataloaders_dict = create_dataloaders(train_img_dir, transform=train_transforms)

# Nombre de classes basat en els personatges únics
unique_characters_list = [...]  # Aquí s'hauria de calcular o obtenir la llista de personatges únics
num_classes = len(unique_characters_list)

# Inicialitzar el model
model = initialize_model(num_classes)

# Configuració del dispositiu (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Configuració de la pèrdua i l'optimizador
criterion = torch.nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Funció per entrenar el model
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model

# Entrenar el model
model = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=25)
"""

"""
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt
import time
import os
import copy

import cv2
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.optim import Adam
from torch.nn import Linear
from torch.utils.data import DataLoader, ImageFolder, Subset
from sklearn.preprocessing import LabelEncoder

# Funció per convertir imatges a escala de grisos i redimensionar-les
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (224, 224))
    return image

# Funció per inicialitzar el model ResNet18 i modificar la capa final
def initialize_model(num_classes):
    model = resnet18(pretrained=False)
    model.fc = Linear(model.fc.in_features, num_classes)
    return model

# Transformacions per a les dades
train_transforms = transforms.Compose([
    transforms.Lambda(lambda img: preprocess_image(img)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Carregar les dades
train_img_dir = '/handwriting-recognition/train_v2/train'
test_img_dir = '/handwriting-recognition/validation_v2/validation'

# Crear conjunts de dades i cargadors
dataloaders_dict = {}

# Selecció de subconjunts
train_indices = torch.randperm(len(ImageFolder(train_img_dir)))[:3000]
test_indices = torch.randperm(len(ImageFolder(test_img_dir)))[:300]

# Crear subconjunts
train_subset = Subset(ImageFolder(train_img_dir), train_indices)
test_subset = Subset(ImageFolder(test_img_dir), test_indices)

# Crear DataLoader per a cada subconjunt
batch_size = 8
dataloaders_dict['train'] = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
dataloaders_dict['val'] = DataLoader(test_subset, batch_size=batch_size, shuffle=True, num_workers=4)

# Nombre de classes basat en els personatges únics
unique_characters_list = [...]  # Aquí s'hauria de calcular o obtenir la llista de personatges únics
num_classes = len(unique_characters_list)

# Inicialitzar el model
model = initialize_model(num_classes)

# Configuració del dispositiu (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Configuració de la pèrdua i l'optimizador
criterion = torch.nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Funció per entrenar el model
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model
"""