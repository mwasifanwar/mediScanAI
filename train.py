import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime

from app.models.multimodal_model import MediScanModel
from app.utils.data_loader import MultiModalDataLoader
from app.utils.preprocess import MedicalImagePreprocessor
import torchvision.transforms as transforms

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MediScanModel().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs
        )
        
        self.best_accuracy = 0.0
        self.train_losses = []
        self.val_metrics = []
    
    def prepare_data(self):
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        data_loader = MultiModalDataLoader(
            image_dir='data/images',
            csv_path='data/train_data.csv',
            batch_size=self.config.batch_size
        )
        
        return data_loader.get_data_loaders(train_transform, val_transform)
    
    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(self.device)
            disease_labels = batch['disease_label'].to(self.device)
            clinical_data = batch['clinical_data'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images, clinical_data)
            loss = self.criterion(outputs['disease'], disease_labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs['disease'], 1)
            correct_predictions += (predicted == disease_labels).sum().item()
            total_samples += disease_labels.size(0)
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_samples
        
        return epoch_loss, epoch_accuracy
    
    def validate_epoch(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                disease_labels = batch['disease_label'].to(self.device)
                clinical_data = batch['clinical_data'].to(self.device)
                
                outputs = self.model(images, clinical_data)
                loss = self.criterion(outputs['disease'], disease_labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs['disease'], 1)
                correct_predictions += (predicted == disease_labels).sum().item()
                total_samples += disease_labels.size(0)
        
        epoch_loss = running_loss / len(val_loader)
        epoch_accuracy = correct_predictions / total_samples
        
        return epoch_loss, epoch_accuracy
    
    def train(self, epochs=100):
        train_loader, val_loader = self.prepare_data()
        
        for epoch in range(epochs):
            start_time = time.time()
            
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            self.scheduler.step()
            
            epoch_time = time.time() - start_time
            
            print(f'Epoch {epoch+1}/{epochs}')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            print(f'Time: {epoch_time:.2f}s')
            print('-' * 50)
            
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                self.save_checkpoint(epoch, val_acc)
    
    def save_checkpoint(self, epoch, accuracy):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'config': self.config
        }
        
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(checkpoint, f'checkpoints/best_model_epoch_{epoch}_acc_{accuracy:.4f}.pth')
        torch.save(self.model.state_dict(), 'models/best_model.pth')

class ModelConfig:
    def __init__(self):
        self.batch_size = 16
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.epochs = 100

if __name__ == '__main__':
    config = ModelConfig()
    trainer = Trainer(config)
    trainer.train(config.epochs)