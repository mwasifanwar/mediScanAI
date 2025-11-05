import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
import os

class MedicalImageDataset(Dataset):
    def __init__(self, image_dir, csv_path, transform=None, is_training=True):
        self.image_dir = image_dir
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.is_training = is_training
        
        self.disease_mapping = {
            'normal': 0,
            'pneumonia': 1,
            'covid': 2,
            'fracture': 3,
            'cancer': 4
        }
        
        self.severity_mapping = {
            'mild': 0,
            'moderate': 1,
            'severe': 2
        }
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        image_path = os.path.join(self.image_dir, row['image_filename'])
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        disease_label = self.disease_mapping.get(row['disease'], 0)
        severity_label = self.severity_mapping.get(row.get('severity', 'mild'), 0)
        
        clinical_data = self._get_clinical_data(row)
        
        return {
            'image': image,
            'disease_label': disease_label,
            'severity_label': severity_label,
            'clinical_data': clinical_data,
            'image_path': image_path
        }
    
    def _get_clinical_data(self, row):
        clinical_features = []
        
        features = ['age', 'gender', 'temperature', 'bp_systolic', 
                   'bp_diastolic', 'heart_rate', 'respiratory_rate', 
                   'oxygen_saturation', 'white_blood_cells', 'c_reactive_protein']
        
        for feature in features:
            clinical_features.append(float(row.get(feature, 0.0)))
        
        return torch.tensor(clinical_features, dtype=torch.float32)

class MultiModalDataLoader:
    def __init__(self, image_dir, csv_path, batch_size=16, num_workers=4):
        self.image_dir = image_dir
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def get_data_loaders(self, train_transform, val_transform, split_ratio=0.8):
        full_dataset = MedicalImageDataset(
            self.image_dir, self.csv_path, transform=train_transform
        )
        
        train_size = int(split_ratio * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        val_dataset.dataset.transform = val_transform
        val_dataset.dataset.is_training = False
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader