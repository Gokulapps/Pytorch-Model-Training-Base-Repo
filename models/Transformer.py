class Ultimus(nn.Module):
    def __init__(self, in_features, out_features):
        super(Ultimus, self).__init__()
        self.in_features = in_features 
        self.out_features = out_features
        self.fc_Q = nn.Linear(in_features=self.in_features, out_features=self.out_features) # Input = 48, Output = 8
        self.fc_K = nn.Linear(in_features=self.in_features, out_features=self.out_features) # Input = 48, Output = 8
        self.fc_V = nn.Linear(in_features=self.in_features, out_features=self.out_features) # Input = 48, Output = 8
        self.softmax = nn.Softmax(dim=1)
        self.fc_out = nn.Linear(in_features=self.out_features, out_features=self.in_features) # Input = 8, Output = 48
    
    def forward(self, tensor):
        Q = self.fc_Q(tensor).unsqueeze(2) # 48 --> 8*1
        K = self.fc_K(tensor).unsqueeze(2) # 48 --> 8*1
        V = self.fc_V(tensor).unsqueeze(2) # 48 --> 8*1
        AM = self.softmax(torch.matmul(K, torch.transpose(Q, dim0=1, dim1=2))/(Q.shape[1]**0.5)) # (8*1) * (1*8) --> 8*8
        Z = torch.matmul(AM, V) # (8*8) * (8*1) --> 8*1 
        Z = Z.view(-1, 8) # 8*1 --> 8
        
        return self.fc_out(Z) # 8 --> 48

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Sequential(
                         nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, bias=False), 
                         nn.BatchNorm2d(16), 
                         nn.ReLU()) # Input = 32*32*3, Output = 32*32*16
        self.conv2 = nn.Sequential(
                         nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, bias=False), 
                         nn.BatchNorm2d(32), 
                         nn.ReLU()) # Input = 32*32*16, Output = 32*32*32
        self.conv3 = nn.Sequential(
                         nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, padding=1, bias=False), 
                         nn.BatchNorm2d(48), 
                         nn.ReLU()) # Input = 32*32*32, Output = 32*32*48
        self.gap = nn.AvgPool2d(32) # Input = 32*32*48, Output = 1*1*48
        self.ultimus1 = Ultimus(48, 8) # Input = 48, Output = 48
        self.ultimus2 = Ultimus(48, 8) # Input = 48, Output = 48
        self.ultimus3 = Ultimus(48, 8) # Input = 48, Output = 48
        self.ultimus4 = Ultimus(48, 8) # Input = 48, Output = 48
        self.output = nn.Linear(in_features=48, out_features=10) # Input = 48, Output = 10
        
    def forward(self, tensor):
        # input = 32*32*3
        x = self.conv1(tensor) # 32*32*3 --> 32*32*16
        x = self.conv2(x) # 32*32*16 --> 32*32*32
        x = self.conv3(x) # 32*32*32 --> 32*32*48
        x = self.gap(x) # 32*32*48 --> 1*1*48
        x = x.view(-1, 48) # 1*1*48 --> 48
        x = self.ultimus1(x) # 48 --> 48
        x = self.ultimus2(x) # 48 --> 48
        x = self.ultimus3(x) # 48 --> 48
        x = self.ultimus4(x) # 48 --> 48
        x = self.output(x) # 48 --> 10
        
        return F.softmax(x, dim=1) 
