import torch
import torch.nn as nn
import modules

class QuantumCNN(nn.Module):
    def __init__(self, dataset, out_channels=1, in_channels=1, vqc_layers=3, device='cpu'):
        super(QuantumCNN, self).__init__()
        self.dataset = dataset
        self.quantum_layer = modules.QuantumConvolution(out_channels=out_channels, in_channels=in_channels, vqc_layers=vqc_layers, device=device)
        self.conv_layer2 = nn.Conv2d(in_channels=1,out_channels=4,kernel_size=2).to(device=device)
        self.fc1 = nn.Linear(99*13*4, 1).to(device=device)
        self.pool = nn.MaxPool2d((2,2))
        self.relu = nn.ReLU()

    def forward(self, x): 

        x = self.quantum_layer(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv_layer2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x,1)
        x = self.fc1(x)

        if x.dim() > 1:
            x = x.squeeze(1)
        return x

class ClassicalCNN(nn.Module):
    def __init__(self, dataset, in_channels=1, out_channels=1, kernel_size=2, transfer=False, device='cpu'):
        super(ClassicalCNN, self).__init__()
        self.transfer = transfer
        self.dataset = dataset
        
        if self.transfer:
            
            # If using weights that were pretained from a quantum filter,
            # a custom Conv2D method must be used to account for the
            # weight normalization from existing in a quantum state
            self.conv_layer = modules.FilterNormalizationConv2DParallel(in_channels=in_channels,out_channels=out_channels,kernel_size=2).to(device=device)
        else:
            self.conv_layer = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=2).to(device=device)
        
        self.conv_layer2 = nn.Conv2d(in_channels=1,out_channels=4,kernel_size=kernel_size).to(device=device)
        self.fc1 = nn.Linear(99*13*4, 1).to(device=device)
        self.pool = nn.MaxPool2d((2,2))
        self.relu = nn.ReLU()

    def forward(self, x): 

        x = self.conv_layer(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv_layer2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x,1)
        x = self.fc1(x)

        if x.dim() > 1:
            x = x.squeeze(1)
        return x
