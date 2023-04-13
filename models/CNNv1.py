import torch.nn as nn
import torch.nn.functional as F

class CNNv1(nn.Module):
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.conv_1 = nn.Conv2d(1, 32, 3, padding=1) 
        self.maxpool_1 = nn.MaxPool2d(2)
        
        self.conv_2 = nn.Conv2d(32, 32, 3, padding=1)   
        self.maxpool_2 = nn.MaxPool2d(2)
        
        self.conv_3 = nn.Conv2d(32, 64, 3, padding=1)
        self.maxpool_3 = nn.MaxPool2d(2)
        
        self.conv_4 = nn.Conv2d(64, 64, 3, padding=1)
        self.maxpool_4 = nn.MaxPool2d(2)
        
        # Flatten first...
        self.flatten = nn.Flatten()
        
        # Fully connected layers:
        self.fc_1 = nn.Linear(64 * 64, 64)
        self.dropout = nn.Dropout(p=0.5)
        self.fc_2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.maxpool_1(x)
        x = F.relu(x)
        
        x = self.conv_2(x)
        x = self.maxpool_2(x)
        x = F.relu(x)

        x = self.conv_3(x)
        x = self.maxpool_3(x)
        x = F.relu(x)
        
        x = self.conv_4(x)
        x = self.maxpool_4(x)
        x = F.relu(x)
        
        x = self.flatten(x)
        x = F.relu(self.fc_1(x))
        x = self.dropout(x)
        x = F.relu(self.fc_2(x))
        
        return x
    