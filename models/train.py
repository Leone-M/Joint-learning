from ignite.metrics import PSNR, SSIM, Fbeta, Accuracy, Precision, Recall
from tqdm import tqdm
from torch import nn
import torch
import time
import os

def train_loop(model, dataloader, optim, enc_crit, dec_crit, lmbd, device="cpu"):
  num_batches =  len(dataloader)
  
  model.train()
  
  total_enc_loss = 0.0
  total_dec_loss = 0.0
  
  psnr_metric = PSNR(data_range=1.0)
  ssim_metric = SSIM(data_range=1.0)
  accuracy_metric = Accuracy()
  # average='macro' чтобы была средняя метрика по всем классам, а не по каждому классу отдельно
  precision_metric = Precision(average='macro')
  recall_metric = Recall(average='macro')
  # среднюю по всем батчам, а бета=1 даёт нам f1
  f1_score_metric = Fbeta(beta=1.0, average=True)
  
  for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc=f"Training")):
    optim.zero_grad()
    
    data, target = data.to(device), target.to(device)
    # проходим вперёд
    enc_out, dec_out = model(data)
    
    # чтобы не вычислять граф обратного распространения
    # with torch.no_grad():
    #   encoded_data = enc_out.clone().detach()
    
    # вычисляем потери
    enc_loss = enc_crit.forward(enc_out, target)
    # тут сравниваем с начальными данными т.к. реконструкция
    dec_loss = dec_crit.forward(dec_out, data)
    
    # обновляем метрики
    total_enc_loss += enc_loss.item()
    total_dec_loss += dec_loss.item()
    
    # общий лосс
    total_loss = enc_loss + lmbd * dec_loss
    
    psnr_metric.update((dec_out, data))
    ssim_metric.update((dec_out, data))
    accuracy_metric.update((enc_out, target))
    precision_metric.update((enc_out, target))
    recall_metric.update((enc_out, target))
    f1_score_metric.update((enc_out, target))
    
    # обратное распространение и оптимизация
    total_loss.backward()
    
    optim.step()
  
  # конечное вычисление потерь и метрик
  total_enc_loss = total_enc_loss/num_batches
  total_dec_loss = total_dec_loss/num_batches
  psnr_value = psnr_metric.compute()
  ssim_value = ssim_metric.compute()
  accuracy_value = accuracy_metric.compute()
  precision_value = precision_metric.compute()
  recall_value = recall_metric.compute()
  f1_score_value = f1_score_metric.compute()
  
  # почистим
  psnr_metric.reset()
  ssim_metric.reset()
  accuracy_metric.reset()
  precision_metric.reset()
  recall_metric.reset()
  f1_score_metric.reset()
  
  return total_enc_loss, total_dec_loss, psnr_value, ssim_value,\
         accuracy_value, precision_value, recall_value, f1_score_value

def test_loop(model, enc_crit, dec_crit,  dataloader, device):
  num_batches = len(dataloader)
  
  model.eval()
  
  # метрики теже
  total_enc_loss = 0.0
  total_dec_loss = 0.0
  
  psnr_metric = PSNR(data_range=1.0)
  ssim_metric = SSIM(data_range=1.0)
  accuracy_metric = Accuracy()
  precision_metric = Precision(average='macro')
  recall_metric = Recall(average='macro')
  f1_score_metric = Fbeta(beta=1.0, average=True)
  
  for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc='Test')):
    data, target = data.to(device), target.to(device)
    # без распространения ошибки
    with torch.no_grad():
      enc_out, dec_out = model(data)
      
      # потери
      enc_loss = enc_crit(enc_out, target)
      dec_loss = dec_crit(dec_out, data)
      total_enc_loss += enc_loss.item()
      total_dec_loss += dec_loss.item()
      # метрики
      psnr_metric.update((dec_out, data))
      ssim_metric.update((dec_out, data))
      accuracy_metric.update((enc_out, target))
      precision_metric.update((enc_out, target))
      recall_metric.update((enc_out, target))
      f1_score_metric.update((enc_out, target))
  
  # конечное вычисление потерь и метрик
  total_enc_loss = total_enc_loss/num_batches
  total_dec_loss = total_dec_loss/num_batches
  psnr_value = psnr_metric.compute()
  ssim_value = ssim_metric.compute()
  accuracy_value = accuracy_metric.compute()
  precision_value = precision_metric.compute()
  recall_value = recall_metric.compute()
  f1_score_value = f1_score_metric.compute()
  
  # почистим
  psnr_metric.reset()
  ssim_metric.reset()
  accuracy_metric.reset()
  precision_metric.reset()
  recall_metric.reset()
  f1_score_metric.reset()
  
  return total_enc_loss, total_dec_loss, psnr_value, ssim_value,\
         accuracy_value, precision_value, recall_value, f1_score_value

def model_train(model, train_loader, test_loader, learning_rate, lmbd, betas,
                start_from=None, EPOCHS=50, model_name="", device="cpu") -> tuple[list[float], list[float], list[float], list[float], list[float], list[float], list[float]]:
  # тренировочные потери
  enc_loss_hist = []
  dec_loss_hist = []

  # тестовые потери
  enc_loss_test_hist = []
  dec_loss_test_hist = []

  # тренировочные метрики
  dec_psnr_hist = []
  dec_ssim_hist = []
  accuracy_hist = []
  precision_hist = []
  recall_hist = []
  f1_score_hist = []

  # тестовые метрики
  dec_psnr_test_hist = []
  dec_ssim_test_hist = []
  accuracy_test_hist = []
  precision_test_hist = []
  recall_test_hist = []
  f1_score_test_hist = []
  
  best_PSNR = 0
  best_checkpoint_path = f"./checkpoints_{model_name}/BEST_CHECKPOINT_LMBD{lmbd}_LR{learning_rate}_BETAS{betas}.pth"
  
  enc_criterion = nn.CrossEntropyLoss()
  # MAE даёт менее смазанную картинку, но не так точно как MSE
  dec_criterion = nn.L1Loss()

  optim = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas)
  
  checkpoint_path = f'./checkpoints_{model_name}/{start_from}_CHECKPOINT_LMBD{lmbd}_LR{learning_rate}_BETAS{betas}.pth'
  if start_from is not None and os.path.exists(checkpoint_path):
    
    # Загрузка чекпоинта при его наличии
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    
    enc_loss_hist = checkpoint['enc_loss_hist']
    dec_loss_hist = checkpoint['dec_loss_hist']
    dec_psnr_hist = checkpoint['dec_psnr_hist']
    dec_ssim_hist = checkpoint['dec_ssim_hist']
    accuracy_hist = checkpoint['accuracy_hist']
    precision_hist = checkpoint['precision_hist']
    recall_hist = checkpoint['recall_hist']
    f1_score_hist = checkpoint['f1_score_hist']
    
    enc_loss_test_hist = checkpoint['enc_loss_test_hist']
    dec_loss_test_hist = checkpoint['dec_loss_test_hist']
    dec_psnr_test_hist = checkpoint['dec_psnr_test_hist']
    dec_ssim_test_hist = checkpoint['dec_ssim_test_hist']
    accuracy_test_hist = checkpoint['accuracy_test_hist']
    precision_test_hist = checkpoint['precision_test_hist']
    recall_test_hist = checkpoint['recall_test_hist']
    f1_score_test_hist = checkpoint['f1_score_test_hist']
    
    lmbd = checkpoint['lmbd']
    print(f"Loading from checkpoint at epoch {start_from}")
  
  for epoch in range(start_from if start_from is not None else 1, EPOCHS+1):
    start_time = time.time()
    
    print(f"Epoch: {epoch}/{EPOCHS}")
    enc_loss, dec_loss,\
    dec_psnr_train, dec_ssim_train,\
    accuracy_train, precision_train,\
    recall_train, f1_score_train = train_loop(model, train_loader,
                                              optim, enc_criterion, dec_criterion, lmbd, device)
    
    enc_loss_test, dec_loss_test,\
    dec_psnr_test, dec_ssim_test,\
    accuracy_test, precision_test,\
    recall_test, f1_score_test = test_loop(model, enc_criterion,
                                           dec_criterion,test_loader, device)
    
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    if epoch % 5 == 0:
      print(f"\nTime elapsed: {elapsed_time:.2f}s\n")
      
      # выводим потери
      print(f"Train Enc Loss: {enc_loss:.4f}, Train Dec Loss: {dec_loss:.4f}")
      print(f"Test Enc Loss: {enc_loss_test:.4f}, Test Dec Loss: {dec_loss_test:.4f}")
      # выводим метрики декодера
      print(f"Train PSNR: {dec_psnr_train:.2f} dB, Train SSIM: {dec_ssim_train:.4f}")
      print(f"Test PSNR: {dec_psnr_test:.2f} dB, Test SSIM: {dec_ssim_test:.4f}")
      # выводим метрики енкодера
      print(f"Train Accuracy : {accuracy_train:.4f}, Train Precision: {precision_train:.4f}")
      print(f"Train Recall : {recall_train:.4f}, Train F1 Score: {f1_score_train:.4f}")
      print(f"Test Accuracy : {accuracy_test:.4f}, Test Precision: {precision_test:.4f}")
      print(f"Test Recall : {recall_test:.4f}, Test F1 Score: {f1_score_test:.4f}")
    
    enc_loss_hist.append(enc_loss)
    dec_loss_hist.append(dec_loss)
    dec_psnr_hist.append(dec_psnr_train)
    dec_ssim_hist.append(dec_ssim_train)
    accuracy_hist.append(accuracy_train)
    precision_hist.append(precision_train)
    recall_hist.append(recall_train)
    f1_score_hist.append(f1_score_train)
    
    enc_loss_test_hist.append(enc_loss_test)
    dec_loss_test_hist.append(dec_loss_test)
    dec_psnr_test_hist.append(dec_psnr_test)
    dec_ssim_test_hist.append(dec_ssim_test)
    accuracy_test_hist.append(accuracy_test)
    precision_test_hist.append(precision_test)
    recall_test_hist.append(recall_test)
    f1_score_test_hist.append(f1_score_test)
    
    if dec_psnr_test > best_PSNR:
      best_PSNR= dec_psnr_test
      torch.save({
        'epoch': epoch,
        
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        
        'enc_loss_hist': enc_loss_hist,
        'dec_loss_hist': dec_loss_hist,
        'dec_psnr_hist': dec_psnr_hist,
        'dec_ssim_hist': dec_ssim_hist,
        'accuracy_hist': accuracy_hist,
        'precision_hist': precision_hist,
        'recall_hist': recall_hist,
        'f1_score_hist': f1_score_hist,
        
        'enc_loss_test_hist': enc_loss_test_hist,
        'dec_loss_test_hist': dec_loss_test_hist,
        'dec_psnr_test_hist': dec_psnr_test_hist,
        'dec_ssim_test_hist': dec_ssim_test_hist,
        'accuracy_test_hist': accuracy_test_hist,
        'precision_test_hist': precision_test_hist,
        'recall_test_hist': recall_test_hist,
        'f1_score_test_hist': f1_score_test_hist,
        
        'lmbd': lmbd,
        'lr': learning_rate,
        'betas': betas
    }, f=best_checkpoint_path)
    
    # делаем чекпоинт в state_dict
    if epoch % 5 == 0:
      torch.save({
        'epoch': epoch,
        
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        
        'enc_loss_hist': enc_loss_hist,
        'dec_loss_hist': dec_loss_hist,
        'dec_psnr_hist': dec_psnr_hist,
        'dec_ssim_hist': dec_ssim_hist,
        'accuracy_hist': accuracy_hist,
        'precision_hist': precision_hist,
        'recall_hist': recall_hist,
        'f1_score_hist': f1_score_hist,
        
        'enc_loss_test_hist': enc_loss_test_hist,
        'dec_loss_test_hist': dec_loss_test_hist,
        'dec_psnr_test_hist': dec_psnr_test_hist,
        'dec_ssim_test_hist': dec_ssim_test_hist,
        'accuracy_test_hist': accuracy_test_hist,
        'precision_test_hist': precision_test_hist,
        'recall_test_hist': recall_test_hist,
        'f1_score_test_hist': f1_score_test_hist,
        
        'lmbd': lmbd,
        'lr': learning_rate,
        'betas': betas
    }, f=f'./checkpoints_{model_name}/{epoch}_CHECKPOINT_LMBD{lmbd}_LR{learning_rate}_BETAS{betas}.pth')
  return enc_loss_hist, enc_loss_test_hist, dec_loss_hist, dec_loss_test_hist,\
         dec_psnr_hist, dec_ssim_hist, accuracy_hist, precision_hist, recall_hist,\
         f1_score_hist, dec_psnr_test_hist, dec_ssim_test_hist, accuracy_test_hist,\
         precision_test_hist, recall_test_hist, f1_score_test_hist
