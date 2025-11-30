import matplotlib.pyplot as plt
from main import Models
import os
from load_model import load_model

def plot_metrics(model_path):
  # Define the path to the checkpoint file
  current_model_type = Models.CNN
  
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

  # Plot the metrics
  plt.figure(figsize=(12, 10))

  plt.subplot(3, 2, 1)
  plt.plot(enc_loss_hist, label='Encoder Loss')
  plt.plot(dec_loss_hist, label='Decoder Loss')
  plt.legend()
  plt.title('Training Loss')

  plt.subplot(3, 2, 2)
  plt.plot(dec_psnr_hist, label='PSNR')
  plt.legend()
  plt.title('PSNR')

  plt.subplot(3, 2, 3)
  plt.plot(dec_ssim_hist, label='SSIM')
  plt.legend()
  plt.title('SSIM')

  plt.subplot(3, 2, 4)
  plt.plot(accuracy_hist, label='Accuracy')
  plt.legend()
  plt.title('Accuracy')

  plt.subplot(3, 2, 5)
  plt.plot(precision_hist, label='Precision')
  plt.legend()
  plt.title('Precision')

  plt.subplot(3, 2, 6)
  plt.plot(recall_hist, label='Recall')
  plt.legend()
  plt.title('Recall')

  plt.tight_layout()
  
  out_path = os.path.join(subdir, f'{os.path.basename(model_path)[:-4]}.png')
  plt.savefig(out_path)
  
  plt.show()

plot_metrics(model_path=
             './checkpoints_conv/75_CHECKPOINT_LMBD0.5_LR0.001_BETAS(0.85, 0.9).pth')