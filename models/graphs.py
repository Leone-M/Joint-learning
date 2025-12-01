import matplotlib.pyplot as plt
from main import Models
import numpy as np
import os
from load_model import load_model

def plot_metrics(model_path, model_type: Models):
  # Define the path to the checkpoint file
  current_model_type = model_type
  
  # Load the model and metrics
  model, optim, enc_loss_hist, dec_loss_hist,\
    dec_psnr_hist, dec_ssim_hist, accuracy_hist,\
      precision_hist, recall_hist, f1_score_hist,\
        enc_loss_test_hist, dec_loss_test_hist,\
          dec_psnr_test_hist, dec_ssim_test_hist,\
            accuracy_test_hist, precision_test_hist,\
              recall_test_hist, f1_score_test_hist, lmbd, betas, lr = load_model(model_path, current_model_type)

  out_dir = 'funny_pictures'
  os.makedirs(out_dir, exist_ok=True)
  subdir = os.path.join(out_dir, f"{current_model_type.value}_{lmbd}_{lr}_{betas}")
  os.makedirs(subdir, exist_ok=True)

  num_epochs = len(enc_loss_hist)
  xticks = np.linspace(0, num_epochs, 15, dtype=int)

  # Plot the metrics
  plt.figure(figsize=(12, 10))

  plt.subplot(3, 2, 1)
  plt.plot(enc_loss_hist, label='Encoder Loss')
  plt.plot(dec_loss_hist, label='Decoder Loss')
  plt.xticks(xticks)
  plt.yticks(np.linspace(min(enc_loss_hist + dec_loss_hist), max(enc_loss_hist + dec_loss_hist), 10))
  plt.legend()
  plt.title('Training Loss')

  plt.subplot(3, 2, 2)
  plt.plot(dec_psnr_hist, label='PSNR')
  plt.xticks(xticks)
  plt.yticks(np.linspace(min(dec_psnr_hist), max(dec_psnr_hist), 10))
  plt.legend()
  plt.title('PSNR')

  plt.subplot(3, 2, 3)
  plt.plot(dec_ssim_hist, label='SSIM')
  plt.xticks(xticks)
  plt.yticks(np.linspace(min(dec_ssim_hist), max(dec_ssim_hist), 10))
  plt.legend()
  plt.title('SSIM')

  plt.subplot(3, 2, 4)
  plt.plot(accuracy_hist, label='Accuracy')
  plt.xticks(xticks)
  plt.yticks(np.linspace(min(accuracy_hist), max(accuracy_hist), 10))
  plt.legend()
  plt.title('Accuracy')

  plt.subplot(3, 2, 5)
  plt.plot(precision_hist, label='Precision')
  plt.xticks(xticks)
  plt.yticks(np.linspace(min(precision_hist), max(precision_hist), 10))
  plt.legend()
  plt.title('Precision')

  plt.subplot(3, 2, 6)
  plt.plot(recall_hist, label='Recall')
  plt.xticks(xticks)
  plt.yticks(np.linspace(min(recall_hist), max(recall_hist), 10))
  plt.legend()
  plt.title('Recall')

  plt.tight_layout()
  
  out_path = os.path.join(subdir, f'{os.path.basename(model_path)[:-4]}.png')
  plt.savefig(out_path)
  
  plt.show()
  
def plot_lambdas(checkpoints: list[str], model_type: Models):
  # Define the path to the checkpoint file
  current_model_type = model_type
  
  if len(checkpoints) != 4:
    raise ValueError("Expected 4 checkpoints for different lambdas")
  
  data = {"0": None, "0.1": None, "0.5": None, "1": None}
  
  # Если чекпоинты не в порядке возрастания лямбда
  # То на запускающего код насылается проклятие
  for (idx, checkpoint) in enumerate(checkpoints):
    # Load the model and metrics
    model, optim, enc_loss_hist, dec_loss_hist,\
      dec_psnr_hist, dec_ssim_hist, accuracy_hist,\
        precision_hist, recall_hist, f1_score_hist,\
          enc_loss_test_hist, dec_loss_test_hist,\
            dec_psnr_test_hist, dec_ssim_test_hist,\
              accuracy_test_hist, precision_test_hist,\
                recall_test_hist, f1_score_test_hist, lmbd, betas, lr = load_model(checkpoint, current_model_type)
                # записываем инфу с чекпоинта под каждую лямбду в словарь
    key = list(data.keys())[idx]
    data[key] = [enc_loss_hist, dec_loss_hist, dec_psnr_hist, dec_ssim_hist, accuracy_hist]

  num_epochs = len(data['0'][0])
  xticks = np.linspace(0, num_epochs, 15, dtype=int)

  out_dir = 'funny_lambdas'
  os.makedirs(out_dir, exist_ok=True)
  subdir = os.path.join(out_dir, f"{current_model_type.value}")
  os.makedirs(subdir, exist_ok=True)
  
  out_path = os.path.join(subdir, 'Lambda_comparison.png')
  
  plt.figure(figsize=(16, 14))
  for idx, (key, metrics) in enumerate(data.items()):
    plt.subplot(5, 4, idx+1)
    plt.plot(metrics[0], label='Enc Loss')
    plt.xticks(xticks)
    plt.yticks(np.linspace(min(metrics[0]), max(metrics[0]), 10))
    plt.legend()
    plt.title(f'ENC_LOSS; LAMBDA = {key}')
    
    plt.subplot(5, 4, idx+5)
    plt.plot(metrics[1], label='Dec Loss')
    plt.xticks(xticks)
    plt.yticks(np.linspace(min(metrics[1]), max(metrics[1]), 10))
    plt.legend()
    plt.title(f'DEC_LOSS; LAMBDA = {key}')
    
    plt.subplot(5, 4, idx+9)
    plt.plot(metrics[2], label='Rec PSNR')
    plt.xticks(xticks)
    plt.yticks(np.linspace(min(metrics[2]), max(metrics[2]), 10))
    plt.legend()
    plt.title(f'REC_PSNR; LAMBDA = {key}')
    
    plt.subplot(5, 4, idx+13)
    plt.plot(metrics[3], label='Rec SSIM')
    plt.xticks(xticks)
    plt.yticks(np.linspace(min(metrics[3]), max(metrics[3]), 10))
    plt.legend()
    plt.title(f'REC_SSIM; LAMBDA = {key}')
    
    plt.subplot(5, 4, idx+17)
    plt.plot(metrics[4], label='Enc accuracy')
    plt.xticks(xticks)
    plt.yticks(np.linspace(min(metrics[4]), max(metrics[4]), 10))
    plt.legend()
    plt.title(f'ENC_ACCURACY; LAMBDA = {key}')
  
  plt.tight_layout()
  plt.savefig(out_path)
  plt.show()

plot_metrics(model_path=
             './checkpoints_conv/75_CHECKPOINT_LMBD0_LR0.002_BETAS(0.9, 0.99).pth',
             model_type=Models.CNN)

checkpoint_comparison = [
  './checkpoints_conv/75_CHECKPOINT_LMBD0_LR0.002_BETAS(0.9, 0.99).pth',
  './checkpoints_conv/75_CHECKPOINT_LMBD0.1_LR0.002_BETAS(0.9, 0.99).pth',
  './checkpoints_conv/75_CHECKPOINT_LMBD0.5_LR0.002_BETAS(0.9, 0.99).pth',
  './checkpoints_conv/75_CHECKPOINT_LMBD1.0_LR0.002_BETAS(0.9, 0.99).pth'
]

plot_lambdas(checkpoint_comparison, Models.CNN)