import os,random
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import numpy as np
from tqdm import tqdm
from evaluation_metrics import *
from attack import *
from fs_ot import *

best_wer = 1000000000

def train(model, device, train_loader, criterion, optimizer, scheduler, text_transform, epoch):
    
    model.train()
    train_loss = 0
    iterator = tqdm(train_loader)
    
    for batch_idx, _data in enumerate(iterator):
        spectrograms, labels, input_lengths, label_lengths = _data 
        spectrograms, labels, model = spectrograms.to(device), labels.to(device), model.to(device)

        optimizer.zero_grad()

        output = model(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1) # (time, batch, n_class)
        
        loss = criterion(output, labels, input_lengths, label_lengths).to(device)
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
        
        #decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths,text_transform)
        
    print("Training Epoch: [{}]  loss: [{:.2f}] \n".format(epoch+1,train_loss/len(train_loader)))
    
    
def test(model, device, test_loader, criterion, epoch, text_transform, filename, modelname):
          
    global best_wer
    print('\nevaluating...')
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    
    with torch.no_grad():
        iterator = tqdm(test_loader)
        for i, _data in enumerate(iterator):
            spectrograms, labels, input_lengths, label_lengths = _data 
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            output = model(spectrograms)  # (batch, time, n_class)
            
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1) # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)
            #test_loss += loss.item() / len(test_loader)
            test_loss += loss.item()
          
            decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths,text_transform)
            for j in range(len(decoded_preds)):
                test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                test_wer.append(wer(decoded_targets[j], decoded_preds[j]))

    avg_cer = sum(test_cer)/len(test_cer)
    avg_wer = sum(test_wer)/len(test_wer)

    
    f = open(filename+".txt","a+")
    f.write('Testing Epoch: [{}] Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(epoch+1, test_loss/len(test_loader), avg_cer, avg_wer))
    f.close()    
    
    print('Testing Epoch: [{}] Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(epoch+1, test_loss/len(test_loader), avg_cer, avg_wer))
          
    
    if avg_wer < best_wer:
        print('Saving Best model...')
        
        if isinstance(model, torch.nn.DataParallel):
            print("multiple GPU")
            state = {
                'model':model.module.state_dict(),
                'model1': model.state_dict(),
                'model2': model,
                'wer': wer,
                'epoch': epoch, 
            }
        else:
            print("not multiple GPU")
            state = {
                    'model':model.state_dict(),
                    'wer':wer,
                    'epoch':epoch,
            }
            
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_point = './checkpoint/'
        if not os.path.isdir(save_point):
            os.mkdir(save_point)
            
        torch.save(state, save_point+modelname+'model.pth.tar')
        best_wer = avg_wer
        
    return best_wer

    
def mixPGD_train(model, device, train_loader, criterion, optimizer, scheduler, eps, iters, alpha, text_transform, epoch):
    
    model.train()
    train_loss = 0
    iterator = tqdm(train_loader)
    
    for batch_idx, _data in enumerate(iterator):
        spectrograms, labels, input_lengths, label_lengths = _data 
        #spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()
        
        output = model(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1) # (time, batch, n_class)
        loss_CTC = criterion(output, labels, input_lengths, label_lengths).to(device)

        x_adv = mixPGD(model,spectrograms, labels, input_lengths, label_lengths,device,eps,iters,alpha,criterion)
        output = model(x_adv)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1) # (time, batch, n_class)
        
        loss_adv = criterion(output, labels, input_lengths, label_lengths).to(device)
        
        loss = loss_CTC+loss_adv
        loss.backward()

        optimizer.step()
        scheduler.step()
        
        train_loss += loss.item()
        
    print("Training Epoch: [{}]  loss: [{:.2f}] \n".format(epoch+1,train_loss/len(train_loader)))    
    
    
def FeaScatter_Train(model, device, train_loader, criterion, optimizer, scheduler, eps, iters, alpha, text_transform, epoch):
    
    model.train()
    train_loss = 0
    iterator = tqdm(train_loader)
    
    for batch_idx, _data in enumerate(iterator):
        spectrograms, labels, input_lengths, label_lengths = _data 
        #spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()
        
        x_adv,loss_fs = FeatureScatter(model,spectrograms,labels,device,input_lengths,label_lengths,eps,iters,alpha,criterion, text_transform)
        loss = loss_fs.mean()

        loss.backward()

        optimizer.step()
        scheduler.step()
        
        train_loss += loss.item()
        
    print("Training Epoch: [{}]  loss: [{:.2f}] \n".format(epoch+1,train_loss/len(train_loader)))

    
def PGD_train(model, device, train_loader, criterion, optimizer, scheduler, eps, iters, alpha, text_transform, epoch):
    
    model.train()
    train_loss = 0
    iterator = tqdm(train_loader)    
    for batch_idx, _data in enumerate(iterator):
        spectrograms, labels, input_lengths, label_lengths = _data 
        optimizer.zero_grad()

        output = model(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1) # (time, batch, n_class)        
        loss_CTC = criterion(output, labels, input_lengths, label_lengths).to(device)

        x_adv = PGD(model,spectrograms, labels, input_lengths, label_lengths,device,eps,iters,alpha,criterion)
        output_adv = model(x_adv)  # (batch, time, n_class)
        output_adv = F.log_softmax(output_adv, dim=2)
        output_adv = output_adv.transpose(0, 1) # (time, batch, n_class)
        
        loss_adv = criterion(output_adv, labels, input_lengths, label_lengths).to(device)
        loss = loss_adv + loss_CTC
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
        
    print("Training Epoch: [{}]  loss: [{:.2f}] \n".format(epoch+1,train_loss/len(train_loader)))

    

def FGSM_train(model, device, train_loader, criterion, optimizer, scheduler, eps, text_transform, epoch):
    
    model.train()
    train_loss = 0
    iterator = tqdm(train_loader)
    
    for batch_idx, _data in enumerate(iterator):
        spectrograms, labels, input_lengths, label_lengths = _data 
        optimizer.zero_grad()
        
        output = model(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1) # (time, batch, n_class)        
        loss_CTC = criterion(output, labels, input_lengths, label_lengths).to(device)
        
        x_adv= FGSM(model,spectrograms, labels, input_lengths, label_lengths,device,eps,criterion)
        output_adv = model(x_adv)  # (batch, time, n_class)
        output_adv = F.log_softmax(output_adv, dim=2)
        output_adv = output_adv.transpose(0, 1) # (time, batch, n_class)
        
        loss_adv = criterion(output_adv, labels, input_lengths, label_lengths).to(device)
        loss = loss_CTC + loss_adv
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
        
    print("Training Epoch: [{}]  loss: [{:.2f}] \n".format(epoch+1,train_loss/len(train_loader)))

    

def FGSM_test(model, device, test_loader, criterion, eps, text_transform,filename):
          
    global best_wer
    print('\nevaluating...')
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    

    iterator = tqdm(test_loader)
    for i, _data in enumerate(iterator):
        spectrograms, labels, input_lengths, label_lengths = _data 
        spectrograms, labels = spectrograms.to(device), labels.to(device)
        adv_inputs = FGSM(model,spectrograms, labels,input_lengths, label_lengths,device,eps,criterion)
        output = model(adv_inputs)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1) # (time, batch, n_class)

        loss = criterion(output, labels, input_lengths, label_lengths)
            #test_loss += loss.item() / len(test_loader)
        test_loss += loss.item()
          
        decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths,text_transform)
        for j in range(len(decoded_preds)):
            test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
            test_wer.append(wer(decoded_targets[j], decoded_preds[j]))


    avg_cer = sum(test_cer)/len(test_cer)
    avg_wer = sum(test_wer)/len(test_wer)

    
    f = open(filename+".txt","a+")
    f.write('FGSM Testing: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.6f}\n'.format(test_loss/len(test_loader), avg_cer, avg_wer))
    f.close()    
    
    print('FGSM Testing: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.6f}\n'.format(test_loss/len(test_loader), avg_cer, avg_wer))

    
    

def PGD_test(model, device, test_loader, criterion, eps, iters, alpha, text_transform,filename):
          
    global best_wer
    print('\nevaluating...')
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    

    iterator = tqdm(test_loader)
    for i, _data in enumerate(iterator):
        spectrograms, labels, input_lengths, label_lengths = _data 
        spectrograms, labels = spectrograms.to(device), labels.to(device)
        adv_inputs = PGD(model,spectrograms, labels, input_lengths, label_lengths,device,eps,iters,alpha,criterion)

        output = model(adv_inputs)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1) # (time, batch, n_class)

        loss = criterion(output, labels, input_lengths, label_lengths)
            #test_loss += loss.item() / len(test_loader)
        test_loss += loss.item()
          
        decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths,text_transform)
        for j in range(len(decoded_preds)):
            test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
            test_wer.append(wer(decoded_targets[j], decoded_preds[j]))


    avg_cer = sum(test_cer)/len(test_cer)
    avg_wer = sum(test_wer)/len(test_wer)

    
    f = open(filename+".txt","a+")
    f.write('PGD Testing: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss/len(test_loader), avg_cer, avg_wer))
    f.close()    
    
    print('PGD Testing: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss/len(test_loader), avg_cer, avg_wer))    
    

    
def data_processing(data, text_transform ,data_type="train"):
    
    train_audio_transforms = nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
        torchaudio.transforms.TimeMasking(time_mask_param=100)
        )

    valid_audio_transforms = torchaudio.transforms.MelSpectrogram()


    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (waveform, _, utterance, _, _, _) in data:
        if data_type == 'train':
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        elif data_type == 'valid':
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        else:
            raise Exception('data_type should be train or valid')
        spectrograms.append(spec)
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0]//2)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths


def GreedyDecoder(output, labels, label_lengths, text_transform, blank_label=28, collapse_repeated=True):
	arg_maxes = torch.argmax(output, dim=2)
	decodes = []
	targets = []
	for i, args in enumerate(arg_maxes):
		decode = []
		targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
		for j, index in enumerate(args):
			if index != blank_label:
				if collapse_repeated and j != 0 and index == args[j -1]:
					continue
				decode.append(index.item())
		decodes.append(text_transform.int_to_text(decode))
	return decodes, targets
          
    
           
