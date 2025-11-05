import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

class ImageEncoder(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super(ImageEncoder, self).__init__()
        
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            self.feature_dim = 2048
        elif backbone == 'densenet121':
            self.backbone = models.densenet121(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            self.feature_dim = 1024
        else:
            raise ValueError("Unsupported backbone")
    
    def forward(self, x):
        features = self.backbone(x)
        return features.view(features.size(0), -1)

class ClinicalDataEncoder(nn.Module):
    def __init__(self, input_dim=10, hidden_dims=[64, 128]):
        super(ClinicalDataEncoder, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        self.output_dim = prev_dim
    
    def forward(self, x):
        return self.network(x)

class MediScanModel(nn.Module):
    def __init__(self, num_classes=5, image_backbone='resnet50'):
        super(MediScanModel, self).__init__()
        
        self.image_encoder = ImageEncoder(backbone=image_backbone)
        self.clinical_encoder = ClinicalDataEncoder()
        
        combined_dim = self.image_encoder.feature_dim + self.clinical_encoder.output_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        self.disease_head = nn.Linear(combined_dim, num_classes)
        self.severity_head = nn.Linear(combined_dim, 3)
    
    def forward(self, image, clinical_data=None):
        image_features = self.image_encoder(image)
        
        if clinical_data is not None:
            clinical_features = self.clinical_encoder(clinical_data)
            combined_features = torch.cat([image_features, clinical_features], dim=1)
        else:
            combined_features = image_features
        
        disease_logits = self.disease_head(combined_features)
        severity_logits = self.severity_head(combined_features)
        
        return {
            'disease': disease_logits,
            'severity': severity_logits,
            'features': combined_features
        }

class AttentionModule(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features):
        attention_weights = self.attention(features)
        return features * attention_weights