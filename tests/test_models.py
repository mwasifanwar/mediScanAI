import torch
import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.models.multimodal_model import MediScanModel, ImageEncoder, ClinicalDataEncoder
from app.models.explainable_ai import ExplainableAI

class TestModels:
    def test_image_encoder(self):
        encoder = ImageEncoder()
        x = torch.randn(2, 3, 224, 224)
        output = encoder(x)
        assert output.shape == (2, 2048)
    
    def test_clinical_encoder(self):
        encoder = ClinicalDataEncoder()
        x = torch.randn(2, 10)
        output = encoder(x)
        assert output.shape == (2, 128)
    
    def test_multimodal_model(self):
        model = MediScanModel()
        image_input = torch.randn(2, 3, 224, 224)
        clinical_input = torch.randn(2, 10)
        
        output = model(image_input, clinical_input)
        
        assert 'disease' in output
        assert 'severity' in output
        assert 'features' in output
        assert output['disease'].shape == (2, 5)
        assert output['severity'].shape == (2, 3)
    
    def test_explainable_ai(self):
        model = MediScanModel()
        explainer = ExplainableAI(model)
        
        input_tensor = torch.randn(1, 3, 224, 224)
        explanation = explainer.generate_explanation(input_tensor)
        
        assert explanation is not None

if __name__ == '__main__':
    pytest.main([__file__])