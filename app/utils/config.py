import yaml
import os

class Config:
    def __init__(self, config_path='config.yaml'):
        self.config_path = config_path
        self._load_config()
    
    def _load_config(self):
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
    
    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, {})
        return value if value != {} else default

class ModelConfig:
    def __init__(self):
        self.image_size = (224, 224)
        self.backbone = 'resnet50'
        self.num_classes = 5
        self.clinical_dim = 10
        self.dropout_rate = 0.3
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5

class TrainingConfig:
    def __init__(self):
        self.batch_size = 16
        self.epochs = 100
        self.patience = 10
        self.warmup_epochs = 5
        self.accumulation_steps = 2

config = Config()
model_config = ModelConfig()
training_config = TrainingConfig()