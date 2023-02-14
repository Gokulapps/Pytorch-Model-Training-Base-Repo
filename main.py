# -*- coding: utf-8 -*-
"""main.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LsEz3UjmDm1mjLymYa0gMTQZqWWkH_7b
"""

import torch 
import torchvision 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
import torch.autograd as grad
import matplotlib.pyplot as plt
import numpy as np
import argparse
import warnings
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from models.resnet import ResNet18 
from utils import *
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Model Training on Selected PyTorch Dataset')
parser.add_argument('dataset', default='CIFAR10', type=str, help='Pytorch Datasets like CIFAR10, MNIST, FashionMNIST')
parser.add_argument('epochs', default=20, type=int, help='Number of Epochs')
parser.add_argument('--resume', '-r', action='store_true',help='Resume from checkpoint')
args = parser.parse_args()
Epochs = args.epochs
dataset_class = getattr(torchvision.datasets, args.dataset)
best_acc = 0
train_loss = []
train_acc = []
test_loss = []
test_acc = []
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ResNet18().to(device)
if device == 'cuda':
  print("=> Parallelizing Training across Multiple GPU's")
  model = torch.nn.DataParallel(model)
  cudnn.benchmark = True

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvisionDataset(root='./data', train=True, download=True, transform=transform)
test_dataset =  torchvisionDataset(root='./data', train=False, download=True, transform=transform)
train_dataset_mean, train_dataset_std = get_mean_and_std(train_dataset, 3)
test_dataset_mean, test_dataset_std = get_mean_and_std(test_dataset, 3)
train_loader = DataLoader(AlbumentationDataset(train_dataset, train_dataset_mean, train_dataset_std, 32, train=True), batch_size=4, shuffle=True, num_workers=2, pin_memory = True)
test_loader = DataLoader(AlbumentationDataset(test_dataset, test_dataset_mean, test_dataset_std, 32, train=False), batch_size=4, shuffle=True, num_workers=2, pin_memory = True)

def train(model, device, train_loader, optimizer, l1_reg):
  global train_acc, train_loss
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  criterion = nn.CrossEntropyLoss()
  for batch_idx, (images, target) in enumerate(pbar):
    images, target = images.to(device), target.to(device)
    optimizer.zero_grad()
    y_pred = model(images)
    loss = criterion(y_pred, target) 
    if l1_reg:
      l1 = 0
      for param in model.parameters():
        l1 += param.abs().sum()
      loss += lambda_y * l1
    train_loss.append(loss)
    loss.backward()
    optimizer.step()  
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(images)
    accuracy = 100*correct/processed
    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={accuracy:0.2f}')
    train_acc.append(accuracy)

def test(model, device, test_loader, epoch):
  global best_acc, test_loss, test_acc
  model.eval()
  loss = 0
  correct = 0
  criterion = nn.CrossEntropyLoss()
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      loss += criterion(output, target).item()  # sum up batch loss
      pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()

  loss /= len(test_loader.dataset)
  test_loss.append(test_loss)
  accuracy = 100. * correct / len(test_loader.dataset)
  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),
      accuracy))
  
  test_acc.append(accuracy)
  if accuracy > best_acc:
      print('Saving..')
      state = {
          'model': model.state_dict(),
          'accuracy': accuracy,
          'epoch': epoch,
          'optimizer': optimizer.state_dict()
      }
      if not os.path.isdir('checkpoint'):
          os.mkdir('checkpoint')
      torch.save(state, './checkpoint/model_progress.pth')
      best_acc = accuracy

def fit_model(model, device, trainloader, testloader, l1=False, l2=False):
  global best_acc, Epochs
  if l2:
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
  else:
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
  scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
  if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/model.pth')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_acc = checkpoint['accuracy']
    start_epoch = checkpoint['epoch']
  else:
    start_epoch = 1
  for epoch in range(start_epoch, Epochs+1):
    print("EPOCH:", epoch)
    train(model, device, trainloader, optimizer, l1)
    scheduler.step()
    test(model, device, testloader, epoch)

def scores():
  return model, train_loss, train_acc, test_loss, test_acc

fit_model(model, device, 10, train_loader, test_loader, False, True)
