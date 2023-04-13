import torch.nn as nn
import torch.nn.functional as F

class CNNv2(nn.Module):
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.conv_1 = nn.Conv2d(1, 32, 3, padding=1) 
        self.batch_norm_1 = nn.BatchNorm2d(32)
        self.maxpool_1 = nn.MaxPool2d(2)
        
        self.conv_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(32)
        self.maxpool_2 = nn.MaxPool2d(2)
        
        self.conv_3 = nn.Conv2d(32, 64, 3, padding=1)
        self.batch_norm_3 = nn.BatchNorm2d(64)
        self.maxpool_3 = nn.MaxPool2d(2)
        
        self.conv_4 = nn.Conv2d(64, 64, 3, padding=1)
        self.batch_norm_4 = nn.BatchNorm2d(64)
        self.maxpool_4 = nn.MaxPool2d(2)
        
        self.conv_5 = nn.Conv2d(64, 128, 3, padding=1)
        self.batch_norm_5 = nn.BatchNorm2d(128)
        self.maxpool_5 = nn.MaxPool2d(2)
        
        self.conv_6 = nn.Conv2d(128, 128, 3, padding=1)
        self.batch_norm_6 = nn.BatchNorm2d(128)
        self.maxpool_6 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()
        
        self.fc_1 = nn.Linear(512, 64)
        self.dropout = nn.Dropout(p=0.5)
        self.fc_2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = self.maxpool_1(x)
        x = F.relu(x)
        
        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        x = self.maxpool_2(x)
        x = F.relu(x)

        x = self.conv_3(x)
        x = self.batch_norm_3(x)
        x = self.maxpool_3(x)
        x = F.relu(x)
        
        x = self.conv_4(x)
        x = self.batch_norm_4(x)
        x = self.maxpool_4(x)
        x = F.relu(x)

        x = self.conv_5(x)
        x = self.batch_norm_5(x)
        x = self.maxpool_5(x)
        x = F.relu(x)

        x = self.conv_6(x)
        x = self.batch_norm_6(x)
        x = self.maxpool_6(x)
        x = F.relu(x)
        
        x = self.flatten(x)
        x = F.relu(self.fc_1(x))
        x = self.dropout(x)
        x = F.relu(self.fc_2(x))
        
        return x
    
