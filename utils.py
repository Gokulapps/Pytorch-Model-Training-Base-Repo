# -*- coding: utf-8 -*-
"""Utils.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1BTSF4w5vz84zXxZVC34KQMRvoNgtLK3v
"""

import torch 
import torchvision 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
import torch.autograd as grad
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
warnings.filterwarnings("ignore")

class torchvisionDataset(dataset_class):
  def __init__(self, root='./data', train=True, download=True, transform=None):
    super().__init__(root=root, train=train, download=download, transform=transform)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    image, label = self.data[index], self.targets[index]
    if self.transform:
      image = self.transform(image=image)['image'] 
    return image, label

class AlbumentationDataset(Dataset):
  def __init__(self, data, dataset_mean, dataset_std, image_size, train=True):
    self.data = data
    self.train= train 
    self.image_size = image_size
    self.avg = sum(dataset_mean)/len(dataset_mean)
    self.train_aug = A.Compose([
                          A.Normalize(dataset_mean, dataset_std),
                          A.HorizontalFlip(), 
                          A.ShiftScaleRotate(), 
                          A.CoarseDropout(max_holes=1, max_height=self.image_size//2, max_width=self.image_size//2, fill_value=self.avg),
                          A.ToGray()])
    self.norm_aug = A.Compose([A.Normalize(dataset_mean, dataset_std)])
  def __len__(self):
    return len(self.data)
  def __getitem__(self, index):
    image, label = self.data[index]
    if self.train:
      self.train_aug(image=np.array(image))['image']
    else:
      self.norm_aug(image=np.array(image))['image']
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    return torch.tensor(image, dtype=torch.float), label


def get_mean_and_std(dataset, no_channels):
  try:
    data = dataset.data
    channels = tuple([i for i in range(no_channels)])
    dataset_mean = np.mean(data, axis=channels, dtype=np.float64)/255.
    dataset_std = np.std(data, axis=channels, dtype=np.float64)/255.
    return tuple(dataset_mean), tuple(dataset_std)
  except Exception as e:
    print(e)

def visualize_images(images, target, classes=None, fig_size=(6.4, 4.8)):
  try:
    plt.figure(figsize=fig_size)
    grid = torchvision.utils.make_grid(images, 12)
    if classes:
      collection = [classes[target[i].item()] for i in range(target.shape[0])]
      plt.title('     '.join(collection))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
  except Exception as e:
    print(e)

def misclassified_images(model, testloader, limit=10):
  incorrect_predictions = []
  count = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      predictions = output.argmax(dim=1, keepdim=True)
      incorrect_indices = predictions.ne(target.view_as(predictions)).nonzero(as_tuple=True)[0]
      for index in incorrect_indices[:limit-count]:
        incorrect_index = index.item()
        incorrect_predictions.append([data[incorrect_index], predictions[incorrect_index], target[incorrect_index]])
        # [Original Data, Model Predictions, Actual Labels] is appended to the Global List 
        count += 1
      if count >= limit:
        return incorrect_predictions
  return incorrect_predictions


def plot_graph(test_losses, test_acc, fig_size=(15,10)):
  try:
    fig, axs = plt.subplots(2, 1, figsize=fig_size)
    axs[0].plot(test_losses, color='green')
    axs[0].set_title("Validation Loss")
    axs[0].set_xlabel('Number of Epoch')
    axs[0].set_ylabel('Loss')
    axs[1].plot(test_acc, color='green')
    axs[1].set_title("Validation Accuracy")
    axs[1].set_xlabel('Number of Epoch')
    axs[1].set_ylabel('Accuracy(%)')
    plt.show()
  except Exception as e:
    print(e)


def plot_misclassified_images(incorrect_predictions, classes, row, col, limit, fig_size=(10, 10)):
  try:
    fig = plt.figure(figsize=fig_size)
    for index in range(1, limit+1):
      plt.subplot(row, col, index)
      plt.axis('off')
      image = (incorrect_predictions[index - 1][0].to('cpu').numpy() / 2) + 0.5
      npimage = np.transpose(image, (1, 2, 0))
      plt.title(f'Model Prediction : {classes[incorrect_predictions[index-1][1]]}, Actual Label : {classes[incorrect_predictions[index-1][2]]}')
      plt.imshow(npimage, cmap='gray_r')
  except Exception as e:
    print(e)

def Gradcam(model, input_tensor, device, target):
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=device)
    targets = [ClassifierOutputTarget(target)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True, eigen_smooth=True)
    grayscale_cam = grayscale_cam[0, :]
    npimg = input_tensor.to('cpu').numpy().transpose(0, 2, 3, 1) / 255. 
    visualization = show_cam_on_image(npimg, grayscale_cam, use_rgb=True)
    return visualization
