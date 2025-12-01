import torch
from torch import nn

class EncoderBlock(nn.Module):
  def __init__(self, in_ch, out_ch, kernel, stride, padding, dropout):
    super().__init__()
    self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel, stride=stride, padding=padding, padding_mode='reflect')
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
  def __init__(self, dropout, shared_fc) -> None:
    super().__init__()
    # готовим перед полносвязным слоем
    self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
    self.fc = shared_fc
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
    
  def forward(self, x):
    x = self.upsample(x)
    x = self.conv(x)
    x = self.norm(x)
    x = self.relu(x)
    return x
  
class DecoderBottleNeck(nn.Module):
  def __init__(self, in_ch, out_ch, shared_fc) -> None:
    super().__init__()
    self.fc = shared_fc
    self.relu = nn.ReLU(inplace=True)
    
  def forward(self, x):
    x = self.fc(x)
    x = self.relu(x)
    # обратное view'у в енкодере
    x = x.view(x.size(0), -1, 1, 1)
    return x

class AutoEncoder(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.shared_fc = nn.Linear(128, 128)
    self.classifier = nn.Linear(128, 10)
    self.encoder = nn.Sequential(
      EncoderBlock(in_ch=1, out_ch=64, kernel=7, stride=2, padding=3, dropout=0.2),
      EncoderBlock(in_ch=64, out_ch=128, kernel=3, stride=2, padding=1, dropout=0.2),
      EncoderHead(dropout=0.2, shared_fc=self.shared_fc)
    )
    self.decoder = nn.Sequential(
      DecoderBottleNeck(in_ch=10, out_ch=128, shared_fc=self.shared_fc),
      DecoderBlock(in_ch=128, out_ch=64, kernel=7, stride=1, padding=1, first_block=True),
      DecoderBlock(in_ch=64, out_ch=32, kernel=5, stride=1, padding=2, scale=2),
      DecoderBlock(in_ch=32, out_ch=1, kernel=3, stride=1, padding=1, scale=2)
    )
    
  def forward(self, x):
    enc_feat = self.encoder(x)
    logits = self.classifier(enc_feat)
    dec_out = self.decoder(enc_feat)
    return logits, dec_out

def conv_model(device):
  conv = AutoEncoder().to(device)
  return conv