import torch
import argparse
import json
from PIL import Image
import os

from app.models.multimodal_model import MediScanModel
from app.models.explainable_ai import ExplainableAI
from app.utils.preprocess import MedicalImagePreprocessor

class MediScanInference:
    def __init__(self, model_path='models/best_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MediScanModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.preprocessor = MedicalImagePreprocessor()
        self.explainer = ExplainableAI(self.model)
        
        self.disease_classes = [
            'Normal', 'Pneumonia', 'COVID-19', 'Fracture', 'Cancer'
        ]
        
        self.severity_classes = ['Mild', 'Moderate', 'Severe']
    
    def predict_single(self, image_path, clinical_data=None):
        image = Image.open(image_path).convert('RGB')
        processed_image = self.preprocessor.process(image).unsqueeze(0).to(self.device)
        
        if clinical_data is not None:
            clinical_data = torch.tensor(clinical_data).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(processed_image, clinical_data)
            
            disease_probs = torch.softmax(outputs['disease'], dim=1)
            severity_probs = torch.softmax(outputs['severity'], dim=1)
            
            disease_pred = torch.argmax(disease_probs, dim=1).item()
            severity_pred = torch.argmax(severity_probs, dim=1).item()
            
            explanation = self.explainer.generate_explanation(processed_image)
        
        return {
            'disease': self.disease_classes[disease_pred],
            'disease_confidence': disease_probs[0][disease_pred].item(),
            'severity': self.severity_classes[severity_pred],
            'severity_confidence': severity_probs[0][severity_pred].item(),
            'all_disease_probabilities': {
                cls: prob.item() for cls, prob in zip(self.disease_classes, disease_probs[0])
            },
            'explanation_image': explanation
        }
    
    def predict_batch(self, image_paths, clinical_data_list=None):
        results = []
        
        for i, image_path in enumerate(image_paths):
            clinical_data = None
            if clinical_data_list and i < len(clinical_data_list):
                clinical_data = clinical_data_list[i]
            
            result = self.predict_single(image_path, clinical_data)
            result['image_path'] = image_path
            results.append(result)
        
        return results

def main():
    parser = argparse.ArgumentParser(description='MediScan AI Inference')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default='models/best_model.pth', help='Path to model weights')
    parser.add_argument('--output', type=str, help='Output file for results')
    
    args = parser.parse_args()
    
    inference_engine = MediScanInference(args.model)
    result = inference_engine.predict_single(args.image)
    
    print("MediScan AI Diagnosis Results:")
    print(f"Disease: {result['disease']} (Confidence: {result['disease_confidence']:.4f})")
    print(f"Severity: {result['severity']} (Confidence: {result['severity_confidence']:.4f})")
    print("\nDisease Probabilities:")
    for disease, prob in result['all_disease_probabilities'].items():
        print(f"  {disease}: {prob:.4f}")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")

if __name__ == '__main__':
    main()