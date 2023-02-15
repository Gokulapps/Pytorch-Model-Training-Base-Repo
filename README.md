# Model Training Using Pytorch 

This Repo Containes Python Code for Training Model on any Torchvision Dataset provided as Argument to the Code
<p align="center">
<img width="300" height="300" src="https://user-images.githubusercontent.com/61132761/219113151-b0cb785c-a1eb-4de4-97f0-ec7e5ed1870d.png">    
</p>

# Contents of the Repo 

Files/Folders  |                                                                 Description                                                                            |
---------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
models         | This Directory is going to be used for storing Custom as well as Pretrained Models which can be used in the main script for training it on any           Torchvision dataset                                                                                                                                                     |
main.py        | Python File where the code for Training the Model on any Torchvision Dataset resides                                                                   |
utils.py       | Additional Python File which has some useful functionalities like Plotting Validation Accuracy, Validation Loss, Displaying Misclassified Images along  with few others                                                                                                                                                         |

* Sample Validation Accuracy and Loss Curve used in this Repo
  ![Model Performance](https://user-images.githubusercontent.com/61132761/219105985-1e04e9d0-28ab-4e33-942b-1aa345723c4b.png)

# Command Line Interface used in the Main Script

* Training can be Initiated by Providing Dataset Name and Number of Epochs as Arguments to the Main File
  main.py CIFAR10 20

* Resume Training from the checkpoint
  main.py --resume
