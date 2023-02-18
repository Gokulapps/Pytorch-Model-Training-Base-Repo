import torch 
import torch.nn as nn 
import torch.nn.functional as F

class CustomResnet(nn.Module):
    def __init__(self):
        super(CustomResnet, self).__init__()
        # PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU
        self.prep_layer = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride = 1, padding=1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU())
        # Layer 1
        # X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
        self.block1 = self.xblock(64, 128, 3, 1, 1)
        # R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]
        self.block2 = self.ResBlock(128, 128, 3, 1, 1)
        # Addition Step Implemented in Forward Method
        # Layer 2 Conv 3x3 [256k] -> MaxPooling2D -> BN -> ReLU
        self.layer2 = self.xblock(128, 256, 3, 1, 1)
        # Layer 3
        # X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
        self.block3 = self.xblock(256, 512, 3, 1, 1)
        # R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
        self.block4 = self.ResBlock(512, 512, 3, 1, 1)
        # Addition Step Implemented in Forward Method
        # MaxPooling with Kernel Size 4
        self.MaxPool = nn.MaxPool2d(4, 4)
        # FC Layer 
        self.fc = nn.Linear(in_features=512, out_features=10)
        
        
    def xblock(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding),
                             nn.MaxPool2d(2, 2),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU())
    
    def ResBlock(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(), 
                             nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU())
    
    def forward(self, tensor):
        x = tensor 
        x = self.prep_layer(x) 
        x = self.block1(x)
        res_block1 = self.block2(x)
        x = x + res_block1
        x = self.layer2(x)
        x = self.block3(x)
        res_block2 = self.block4(x)
        x = x + res_block2
        x = self.MaxPool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = x.view(-1, 10)
        
        return F.softmax(x, dim=-1)
