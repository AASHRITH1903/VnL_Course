'''
    Module contains Trainer used in training and testing processes.
'''

import io
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
from tqdm import tqdm
from torchmetrics.text import BLEUScore

class Trainer:
    def __init__(
        self, 
        model, 
        optimizer, 
        scaler, 
        scheduler, 
        train_loader, 
        test_loader,
        device='cpu'
    ):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.bleu = BLEUScore(n_gram=1)

        self.cur_lr = self.optimizer.param_groups[0]['lr']
        self.epoch = 0
        self.train_loss = []
        self.valid_loss = []
        self.test_result = None
            
    def train_epoch(self):
        self.model.train()
        self.epoch += 1

        total_loss = 0

        loop = tqdm(self.train_loader, total=len(self.train_loader))
        loop.set_description(f'Epoch: {self.epoch} | Loss: ---')
        for batch_idx, (img_emb, cap, att_mask) in enumerate(loop):

            img_emb, cap, att_mask = img_emb.to(self.device), cap.to(self.device), att_mask.to(self.device)

            with torch.cuda.amp.autocast():
                loss = self.model.train_forward(img_emb=img_emb, trg_cap=cap, att_mask=att_mask)
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.3)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.optimizer.zero_grad()

            total_loss += loss.item()

            loop.set_description(f'Epoch: {self.epoch} | Loss: {total_loss / (batch_idx + 1):.3f}')
            loop.refresh()

        self.cur_lr = self.optimizer.param_groups[0]['lr']
        self.train_loss.append(total_loss / (batch_idx + 1))

        self.scheduler.step()
    
    def valid_epoch(self):
        self.model.eval()

        total_loss = 0

        loop = tqdm(self.valid_loader, total=len(self.valid_loader))
        loop.set_description(f'Validation Loss: ---')
        for batch_idx, (img_emb, cap, att_mask) in enumerate(loop):

            img_emb, cap, att_mask = img_emb.to(self.device), cap.to(self.device), att_mask.to(self.device)

            with torch.no_grad():
                with torch.cuda.amp.autocast():

                    loss = self.model.train_forward(img_emb=img_emb, trg_cap=cap, att_mask=att_mask)

                    total_loss += loss.item()
                    
                    loop.set_description(f'Validation Loss: {total_loss / (batch_idx + 1):.3f}')
                    loop.refresh()

        self.valid_loss.append(total_loss / (batch_idx + 1))

    def test_epoch(self):
        self.model.eval()
        gts = []
        preds = []
        bleu_scores = []
        for batch_idx, (_, img_emb, cap) in tqdm(enumerate(self.test_loader), total=len(self.test_loader.dataset)):
            caption, _ = self.model(img_emb.to(self.device))
            if caption:
                bleu_scores.append(self.bleu(caption, cap[0]))
        return np.mean(bleu_scores)

    def get_training_data(self):
        return {
            'train_loss': self.train_loss, 
            'valid_loss': self.valid_loss, 
            'lr': self.cur_lr, 
            'examples': self.test_result    
        }

    def save_ckp(self, ckp_path):
        torch.save(
            {
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'scaler_state_dict': self.scaler.state_dict(),
                'tloss': self.train_loss,
                'vloss': self.valid_loss
            }, 
            ckp_path
        )   

    def _load_ckp(
        self, 
        checkpoint_fpath,
        optimizer=False, 
        scheduler=False, 
        scaler=False, 
        epoch=True, 
        train_loss=False, 
        valid_loss=False, 
        device='cpu'
    ):
        '''
            Loads entire checkpoint from file.
        '''

        checkpoint = torch.load(checkpoint_fpath, map_location=device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        if epoch:
            self.epoch = checkpoint['epoch']

        if train_loss:
            self.train_loss = checkpoint['tloss']

        if valid_loss:
            self.valid_loss = checkpoint['vloss']