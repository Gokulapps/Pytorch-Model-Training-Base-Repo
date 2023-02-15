# Model Training Using Pytorch 

This Repo Containes Python Code for Training Model on any Torchvision Dataset provided as Argument to the Code
![Sample CIFAR10 Dataset]([https://github.com/username/repository/image.png](https://user-images.githubusercontent.com/61132761/219100619-816945e3-a504-4f03-a68e-73bd32ea27b6.png))

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
