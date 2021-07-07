import os
import torch
import torch.nn as nn
import torch.utils.data as data
from prefetch_generator import BackgroundGenerator
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import numpy as np
from model import *
from evaluation_metrics import *
from training_config import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class DataLoaderX(data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

    
def main(learning_rate=5e-4, batch_size=20, epochs=10,
         train_url="train-clean-100", test_url="test-clean"):
    hparams = {
        "n_cnn_layers": 3,
        "n_rnn_layers": 5,
        "rnn_dim": 512,
        "n_class": 29,
        "n_feats": 128,
        "stride": 2,
        "dropout": 0.1,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs
    }

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda:0" if use_cuda else "cpu")

    train_dataset = torchaudio.datasets.LIBRISPEECH("/home/aminul/data1", url=train_url, download=True)
    test_dataset = torchaudio.datasets.LIBRISPEECH("/home/aminul/data1", url=test_url, download=True)

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    text_transform = TextTransform()

    train_loader = DataLoaderX(dataset=train_dataset,
                                   batch_size=hparams['batch_size'],
                                   shuffle=True,
                                   collate_fn=lambda x: data_processing(x, text_transform, 'train'),
                                   **kwargs)
    test_loader = DataLoaderX(dataset=test_dataset,
                                  batch_size=hparams['batch_size'],
                                  shuffle=False,
                                  collate_fn=lambda x: data_processing(x, text_transform, 'valid'),
                                  **kwargs)

    model = SpeechRecognitionModel(
        hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
    ).to(device)

    model = nn.DataParallel(model, device_ids=[0,1,2,3,4])

    optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
    criterion = nn.CTCLoss(blank=28).to(device)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'],
                                              steps_per_epoch=int(len(train_loader))*2,
                                              epochs=hparams['epochs'],
                                              anneal_strategy='linear')

    filename = "fs_new1" 
    f = open(filename + ".txt", "w+")
    f.close()

    for epoch in range(0, epochs):        
        train(model, device, train_loader, criterion, optimizer, scheduler, text_transform, epoch)
        FeaScatter_Train(model, device, train_loader, criterion, optimizer, scheduler,  0.00004,5,0.00001, text_transform, epoch)
        best_wer = test(model, device, test_loader, criterion, epoch, text_transform, filename, 'fs_')

    print("loading best model")    
    checkpoint = torch.load('./checkpoint/fs_model.pth.tar', map_location=device)
    model = checkpoint['model2']
    model = model.module.to(device)
    model = nn.DataParallel(model, device_ids=[0,1,2,3,4])
    
    FGSM_test(model, device, test_loader, criterion, 0.00004, text_transform,filename)
    PGD_test(model,device,test_loader,criterion,0.00004,20,0.00001,text_transform,filename)
    
    
if __name__ == '__main__':
    learning_rate = 5e-4
    batch_size = 10
    epochs = 25
    libri_train_set = "train-clean-100"
    libri_test_set = "test-clean"

    main(learning_rate, batch_size, epochs, libri_train_set, libri_test_set)
