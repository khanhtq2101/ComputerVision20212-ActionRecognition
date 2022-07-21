import torch 
import torch.nn as nn
import torchvision

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SpatialNet(nn.Module):
  def __init__(self, n_classes = 10, n_segments = 3):
    super().__init__()
    self.n_segments = n_segments
    self.n_classes = n_classes
    self.trained_epochs = 0

    self.resnet = torchvision.models.resnet101()
    self.fc = nn.Linear(1000, n_classes)
  
  def forward(self, frames):
    #output shape: batch x n_segment x n_classes
    output = torch.zeros(frames.shape[0], self.n_segments, self.n_classes, device = frames.device)

    for seg_idx in range(self.n_segments):
      x = self.resnet(frames[:, :, seg_idx, :, :])
      x = self.fc(x)
      output[:, seg_idx, :] = x

    #average over segment axis
    output = torch.mean(output, dim = 1)
    
    return output

class TemporalNet(nn.Module):
  def __init__(self, n_classes = 6, n_segments = 3, n_consecutive = 1):
    super().__init__()
    self.n_segments = n_segments
    self.n_classes = n_classes
    self.n_consecutive = n_consecutive
    self.trained_epochs = 0

    self.in_cnn = nn.Conv2d(in_channels = 2*n_consecutive, out_channels= 3, kernel_size= 3)
    self.resnet = torchvision.models.resnet101()
    self.fc = nn.Linear(1000, n_classes)
  
  def forward(self, frames):
    #output shape: batch x n_segment x n_classes
    output = torch.zeros(frames.shape[0], self.n_segments, self.n_classes, device = frames.device)

    for seg_idx in range(self.n_segments):
      x = self.in_cnn(frames[:, :, seg_idx, :, :])
      x = self.resnet(x)
      x = self.fc(x)
      output[:, seg_idx, :] = x

    #average over segment axis
    output = torch.mean(output, dim = 1)
    
    return output
