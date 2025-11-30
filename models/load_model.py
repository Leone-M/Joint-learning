from main import Models
import re
import torch
import os

def load_model(model_path, model_type: Models):  
  if model_type  == Models.CNN:
    from conv import conv_model as model_class
  elif model_type == Models.CNN_RESNET:
    from  conv_skip import unet_model as model_class 
  else: 
    from transformer import trans_model as model_class 
  
  model = model_class(device='cuda')
  
  checkpoint = torch.load(model_path)
  
  lmbd = checkpoint.get('lmbd', 0)
  betas = checkpoint.get('betas', (0.85, 0.9))
  learning_rate = checkpoint.get('lr', 0.001)
  
  optim = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas)
  
  model.load_state_dict(checkpoint['model_state_dict'])
  optim.load_state_dict(checkpoint['optimizer_state_dict'])
  enc_loss_hist = checkpoint.get('enc_loss_hist', [])
  dec_loss_hist = checkpoint.get('dec_loss_hist', [])
  dec_psnr_hist = checkpoint.get('dec_psnr_hist', [])
  dec_ssim_hist = checkpoint.get('dec_ssim_hist', [])
  accuracy_hist = checkpoint.get('accuracy_hist', [])
  precision_hist = checkpoint.get('precision_hist', [])
  recall_hist = checkpoint.get('recall_hist', [])
  f1_score_hist = checkpoint.get('f1_score_hist', [])
  enc_loss_test_hist = checkpoint.get('enc_loss_test_hist', [])
  dec_loss_test_hist = checkpoint.get('dec_loss_test_hist', [])
  dec_psnr_test_hist = checkpoint.get('dec_psnr_test_hist', [])
  dec_ssim_test_hist = checkpoint.get('dec_ssim_test_hist', [])
  accuracy_test_hist = checkpoint.get('accuracy_test_hist', [])
  precision_test_hist = checkpoint.get('precision_test_hist', [])
  recall_test_hist = checkpoint.get('recall_test_hist', [])
  f1_score_test_hist = checkpoint.get('f1_score_test_hist', [])
  

  
  return model, optim, enc_loss_hist, dec_loss_hist, dec_psnr_hist, dec_ssim_hist,\
        accuracy_hist, precision_hist, recall_hist, f1_score_hist, enc_loss_test_hist,\
        dec_loss_test_hist, dec_psnr_test_hist, dec_ssim_test_hist, accuracy_test_hist,\
        precision_test_hist, recall_test_hist, f1_score_test_hist, lmbd, betas, learning_rate