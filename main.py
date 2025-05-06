#%% LIB
import os
import torch
# import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from PIL import Image

import torchinfo
from torchinfo import summary
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Subset

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from utils import (
    ConvertToRGB, get_mean_std, train_epoch, train, score, predict, class_counts
    )

import matplotlib.pyplot as plt
import pandas as pd
import random

g = torch.Generator()
g.manual_seed(42)
batch_size = 16


#%% TRANSFORM
### Speed up with MPS of macbook
if torch.backends.mps.is_available():
    device='mps'
else:
    device='cpu'

### Normalize dataset loader
# Get mean of current loader
transform = transforms.Compose([
    ConvertToRGB(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    ])

classes = os.listdir("data")
dataset = datasets.ImageFolder(
    root="data", transform=transform
    )

class_counts(dataset, "Whole dataset")

dataset_loader = DataLoader(dataset, batch_size=batch_size)
mean, std = get_mean_std(dataset_loader)

# Improve transform -> transform_norm
transform_norm = transforms.Compose([
    ConvertToRGB(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
    ])

dataset_norm = datasets.ImageFolder(
    root="data", transform=transform_norm
    )
class_counts(dataset_norm, 'Whole dataset')

dataset_loader = DataLoader(dataset_norm, batch_size=batch_size)

# Check mean & std after normalization
mean, std = get_mean_std(dataset_loader)
print(f"\nData loader:\nMean: {mean},\nStd: {std}")


#%% SPLIT

train_dataset, val_dataset = random_split(dataset_norm, [0.8, 0.2], generator=g)
print(f"Dataset length of {len(dataset_norm)}")
print(f"Train data length of {len(train_dataset)}, \
make up of {len(train_dataset)/len(dataset)*100:.1f}%")
print(f"Validate data length of {len(val_dataset)}, \
make up of {len(val_dataset)/len(dataset)*100:.1f}%")


train_loader = DataLoader(
    train_dataset, shuffle=True, batch_size=batch_size, generator=g
    )
val_loader = DataLoader(
    val_dataset, shuffle=False, batch_size=batch_size
    )

batch_shape = next(iter(dataset_loader))[0].shape
print("Batch shape:", batch_shape)

#%% CNN structure
torch.manual_seed(42)
torch.mps.manual_seed(42)

# Initial model
model = torch.nn.Sequential()

# Layer Convolution 1
conv1 = torch.nn.Conv2d(
    in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1
    )
max_pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
model.append(conv1)
model.append(torch.nn.ReLU())
model.append(max_pool1)

# Layer Convolution 2
conv2 = torch.nn.Conv2d(
    in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1
    )
max_pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
model.append(conv2)
model.append(torch.nn.ReLU())
model.append(max_pool2)

# Layer Linear
model.append(torch.nn.Flatten())
model.append(torch.nn.Dropout())
linear1 = torch.nn.Linear(
    in_features=100352, out_features=100
    )
model.append(linear1)
model.append(torch.nn.ReLU())
model.append(torch.nn.Dropout())

# Layer Output
output_layer = torch.nn.Linear(100, 2)
model.append(output_layer)

# Summarize model structure
summary(model)

# Move model to MPS
model.to(device)

#%% OPTIMIZER
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#%% TRAINING

epochs = 15

epoch = train(
    model, optimizer, loss_fn, train_loader, val_loader, epochs=epochs,
    device=device
    )

#%% POST-TRAIN ANALYSIS
train_losses, val_losses, train_accuracies, val_accuracies, epoch_count = epoch

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

ax[0].plot(pd.Series(train_losses), label='train_loss')
ax[0].plot(pd.Series(val_losses), label='val_loss')
ax[0].legend()


ax[1].plot(pd.Series(train_accuracies), label='train_acc')
ax[1].plot(pd.Series(val_accuracies), label='val_acc')
ax[1].legend()

plt.show()

#%%
probs = predict(model, val_loader, device=device)
predictions = torch.argmax(probs, dim=1)

# Sample 12 random indices from the test dataset
sample_indices = random.sample(range(len(val_loader.dataset.dataset.samples)), 12)

# Create a grid of 4x3 subplots
fig, axes = plt.subplots(4, 3, figsize=(20, 10))

# Iterate over the sampled indices and plot the corresponding images
for ax, idx in zip(axes.flatten(), sample_indices):
    print(ax, idx)
    
    image_path = val_loader.dataset.dataset.samples[idx][0]
    img = Image.open(image_path)

    # Display the image on the axis
    ax.imshow(img)
    ax.axis('off')

    # Get the predicted class for this image
    predicted_class = val_dataset.dataset.classes[val_dataset.dataset.imgs[idx][1]]
    
    # Set the title of the subplot to the predicted class
    ax.set_title(f"Predicted: {predicted_class}", fontsize=14)

plt.tight_layout()


#%% SAVING MODEL
torch.save(model, os.path.join("model", "model_cnn_v2"))
torch.save(model.state_dict(), os.path.join("model", "state_dict_cnn_v2"))





