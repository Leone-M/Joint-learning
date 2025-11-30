import torch
from torch import nn

class EncoderBlock(nn.Module):
  def __init__(self, in_ch, out_ch, kernel, stride, padding, dropout):
    super().__init__()
    self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, padding=padding, padding_mode='reflect', kernel_size=kernel, stride=stride)
    self.norm = nn.BatchNorm2d(out_ch)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
    x = self.conv(x)
    x = self.norm(x)
    x = self.relu(x)
    x = self.dropout(x)
    return x
    
class EncoderHead(nn.Module):
  def __init__(self, dropout) -> None:
    super().__init__()
    # готовим перед полносвязным слоем
    self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
    self.fc = nn.Linear(128, 10)
    self.dropout = nn.Dropout(dropout)
    self.softmax = nn.Softmax(dim=1)
    
  def forward(self, x):
    x = self.global_avg_pool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    x = self.dropout(x)
    return x
  
class DecoderBlock(nn.Module):
  def __init__(self, in_ch, out_ch, kernel, stride, padding, first_block=False, scale=None) -> None:
    super().__init__()
    if first_block:
      self.upsample = nn.Upsample(size=7, mode='bilinear', align_corners=False)
      self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=stride, padding=padding)
    else:
      self.upsample = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)
      self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel, stride=stride, padding=padding)
    self.norm = nn.BatchNorm2d(out_ch)
    self.relu = nn.ReLU(inplace=True)
    
  def forward(self, x, skip_con=None):
    x = self.upsample(x)
    if skip_con is not None:
      x = x + skip_con
    x = self.conv(x)
    x = self.norm(x)
    x = self.relu(x)
    return x
  
class DecoderBottleNeck(nn.Module):
  def __init__(self, in_ch, out_ch) -> None:
    super().__init__()
    self.fc = nn.Linear(in_ch, out_ch)
    self.relu = nn.ReLU(inplace=True)
    
  def forward(self, x):
    x = self.fc(x)
    x = self.relu(x)
    # обратное view'у в енкодере
    x = x.view(x.size(0), -1, 1, 1)
    return x
  
class UNetWithSkipConnections(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.encoder1 = EncoderBlock(in_ch=1, out_ch=64, kernel=7, stride=2, padding=3, dropout=0.2)
    self.encoder2 = EncoderBlock(in_ch=64, out_ch=128, kernel=3, stride=2, padding=1, dropout=0.2)
    self.encoderHead = EncoderHead(dropout=0.2)
    
    self.decoderHead = DecoderBottleNeck(in_ch=10, out_ch=128)
    self.decoder1 = DecoderBlock(in_ch=128, out_ch=64, kernel=7, stride=1, padding=1, first_block=True)
    self.decoder2 = DecoderBlock(in_ch=64, out_ch=32, kernel=5, stride=1, padding=2, scale=2)
    self.decoder3 = DecoderBlock(in_ch=32, out_ch=1, kernel=3, stride=1, padding=1, scale=2)
    
  def forward(self, x):
    # Encoder
    x = self.encoder1(x)
    skip_1 = x
    x = self.encoder2(x)
    skip_2 = x
    x = self.encoderHead(x)
    enc_out = x
    # Decoder
    x = self.decoderHead(x)
    x = self.decoder1(x, skip_2)
    x = self.decoder2(x, skip_1)
    x = self.decoder3(x)
    return enc_out, x
  
def unet_model(device):
  unet_model = UNetWithSkipConnections().to(device)
  return unet_model