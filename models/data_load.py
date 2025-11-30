import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# качаем датасеты
train_dataset = datasets.FashionMNIST(root='./data',
                                      train=True,
                                      transform=transforms.ToTensor(),
                                      download=True)

test_dataset = datasets.FashionMNIST(root='./data',
                                     train=False,
                                     transform=transforms.ToTensor(),
                                     download=True)

def get_loaders(batch_size):
  # создаем загрузчики данных
  train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True)
  
  test_loader = DataLoader(dataset=test_dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           pin_memory=True)
  
  return train_loader, test_loader