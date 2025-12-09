from train import model_train
from enum import Enum
from torchsummary import summary
from conv import conv_model
from conv_skip import unet_model
from transformer import trans_model
import data_load

LAMBDA = [0, 20.0, 80.0, 150.0]
LR = 0.002
BETAS = (0.9, 0.99)
BATCH_SIZE = 256
EPOCHS = 75

class Models(Enum):
  CNN = 'conv'
  CNN_RESNET = 'skip'
  TRANSFORMER = 'transform'

def main(model: Models):
  # устройство
  device = 'cuda' 
  # Выбрать модель
  chosen_model = model
  
  # Дата-лоадеры
  train_loader, test_loader = data_load.get_loaders(batch_size=BATCH_SIZE)
  
  # Делаем каждую модель и запускаем её на тренировку
  # Со всеми возможными лямбда
  for lmbd in LAMBDA:
    if chosen_model == Models.CNN:
      model = conv_model(device)
      summary(model, (1, 28, 28))
      model_train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        learning_rate=LR,
        device=device,
        lmbd=lmbd,
        betas=BETAS,
        EPOCHS=75,
        model_name=Models.CNN.value,
        start_from=None
      )
    elif chosen_model == Models.CNN_RESNET:
      model = unet_model(device)
      summary(model, (1, 28, 28))
      model_train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        learning_rate=LR,
        device=device,
        lmbd=lmbd,
        betas=BETAS,
        EPOCHS=75,
        model_name=Models.CNN_RESNET.value,
        start_from=None
      )
    else:
      model = trans_model(device)
      summary(model, (1, 28, 28))
      model_train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        learning_rate=LR,
        device=device,
        lmbd=lmbd,
        betas=BETAS,
        EPOCHS=75,
        model_name=Models.TRANSFORMER.value,
        start_from=None
      )
      
if __name__ == "__main__":
  main(Models.TRANSFORMER)