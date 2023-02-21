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
from models.custom_resnet import CustomResnet
from utils import *
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Model Training on Selected PyTorch Dataset')
parser.add_argument('dataset', default='CIFAR10', type=str, help='Provide Pytorch Datasets like CIFAR10, MNIST, FashionMNIST')
parser.add_argument('epochs', default=20, type=int, help='Number of Epochs')
parser.add_argument('batch_size', default=64, type=int, help='Batch Size')
parser.add_argument('lr', default=0.001, type=float, help='Learning Rate')
parser.add_argument('max_lr', default=0.017, type=float, help='Maximum Learning Rate for OneCyclePolicy')
parser.add_argument('--augmentation', action='store_true', help='whether to Perform Augmentation or not')
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
args = parser.parse_args()
Epochs = args.epochs
dataset_class = getattr(torchvision.datasets, args.dataset)
best_acc = 0
train_loss = []
train_acc = []
test_loss = []
test_acc = []
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = ResNet18().to(device)
model = CustomResnet().to(device)
if device == 'cuda':
  print("=> Parallelizing Training across Multiple GPU's")
  model = torch.nn.DataParallel(model)

class torchvisionDataset(dataset_class):
  def __init__(self, root='./data', train=True, download=True, transform=None):
    super().__init__(root=root, train=train, download=download, transform=transform)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    image, label = self.data[index], self.targets[index]
    return image, label


transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvisionDataset(root='./data', train=True, download=True, transform=transform)
test_dataset =  torchvisionDataset(root='./data', train=False, download=True, transform=transform)
train_dataset_mean, train_dataset_std = get_mean_and_std(train_dataset, 3)
test_dataset_mean, test_dataset_std = get_mean_and_std(test_dataset, 3)
if args.augmentation:
    train_loader = DataLoader(AlbumentationDataset(train_dataset, train_dataset_mean, train_dataset_std, 32, train=True), batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory = True)
else:
    train_loader = DataLoader(AlbumentationDataset(train_dataset, train_dataset_mean, train_dataset_std, 32, train=False), batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory = True)
test_loader = DataLoader(AlbumentationDataset(test_dataset, test_dataset_mean, test_dataset_std, 32, train=False), batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory = True)


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
      lambda_y = 0.000025
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
  test_loss.append(loss)
  accuracy = 100. * correct / len(test_loader.dataset)
  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
      loss, correct, len(test_loader.dataset), accuracy))
  
  test_acc.append(accuracy)
  if accuracy > best_acc:
      state = {
          'model': model.state_dict(),
          'accuracy': accuracy,
          'epoch': epoch,
          #'optimizer': optimizer.state_dict()
      }
      if not os.path.isdir('checkpoint'):
          os.mkdir('checkpoint')
      torch.save(state, './checkpoint/model_state.pth')
      torch.save(model, './checkpoint/model.pth')
      best_acc = accuracy

def fit_model(model, device, trainloader, testloader, l1=False, l2=False):
  global best_acc, Epochs, test_loss, test_acc
  if l2:
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
  else:
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
  total_steps = len(train_loader)*Epochs
  step_size_up = len(train_loader) * int(round(Epochs * 0.20833333333))
  step_size_down = total_steps - step_size_up
  scheduler = OneCycleLR(optimizer, max_lr=args.max_lr, epochs=Epochs, total_steps=total_steps, steps_per_epoch=len(train_loader), pct_start=step_size_up, anneal_strategy = 'linear')
  if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/model.pth')
    model.load_state_dict(checkpoint['model'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    best_acc = checkpoint['accuracy']
    start_epoch = checkpoint['epoch']
  else:
    start_epoch = 1
  print('Model Training...')
  for epoch in range(start_epoch, Epochs+1):
    print("EPOCH:", epoch)
    train(model, device, trainloader, optimizer, l1)
    scheduler.step()
    test(model, device, testloader, epoch)

fit_model(model, device, train_loader, test_loader, True, False)
print('Model Saved')
print('Plotting Graphs')
plot_graph(test_loss, test_acc, fig_size=(15,10))
print('Displaying Sample Images from Dataset')
images, target = next(iter(train_loader))
visualize_images(images, target, classes=None, fig_size=(20, 10))
print('Main File Completed!!!')
