from flask import Blueprint, render_template, request, jsonify, send_file
import torch
import io
import base64
from PIL import Image
import numpy as np

from app.models.multimodal_model import MediScanModel
from app.models.explainable_ai import ExplainableAI
from app.utils.preprocess import MedicalImagePreprocessor

bp = Blueprint('main', __name__)

model = None
explainer = None
preprocessor = None

def load_model():
    global model, explainer, preprocessor
    model = MediScanModel()
    model.load_state_dict(torch.load('models/best_model.pth', map_location='cpu'))
    model.eval()
    explainer = ExplainableAI(model)
    preprocessor = MedicalImagePreprocessor()

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        image = Image.open(file.stream).convert('RGB')
        processed_image = preprocessor.process(image)
        
        with torch.no_grad():
            predictions = model(processed_image.unsqueeze(0))
            explanation = explainer.generate_explanation(processed_image.unsqueeze(0))
        
        result = {
            'predictions': predictions.numpy().tolist(),
            'explanation': explanation,
            'confidence': torch.max(torch.softmax(predictions, dim=1)).item()
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@bp.route('/batch_predict', methods=['POST'])
def batch_predict():
    files = request.files.getlist('files')
    results = []
    
    for file in files:
        try:
            image = Image.open(file.stream).convert('RGB')
            processed_image = preprocessor.process(image)
            
            with torch.no_grad():
                predictions = model(processed_image.unsqueeze(0))
            
            results.append({
                'filename': file.filename,
                'predictions': predictions.numpy().tolist(),
                'confidence': torch.max(torch.softmax(predictions, dim=1)).item()
            })
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e)
            })
    
    return jsonify({'results': results})

load_model()from flask import Blueprint, render_template, request, jsonify, send_file
import torch
import io
import base64
from PIL import Image
import numpy as np

from app.models.multimodal_model import MediScanModel
from app.models.explainable_ai import ExplainableAI
from app.utils.preprocess import MedicalImagePreprocessor

bp = Blueprint('main', __name__)

model = None
explainer = None
preprocessor = None

def load_model():
    global model, explainer, preprocessor
    model = MediScanModel()
    model.load_state_dict(torch.load('models/best_model.pth', map_location='cpu'))
    model.eval()
    explainer = ExplainableAI(model)
    preprocessor = MedicalImagePreprocessor()

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        image = Image.open(file.stream).convert('RGB')
        processed_image = preprocessor.process(image)
        
        with torch.no_grad():
            predictions = model(processed_image.unsqueeze(0))
            explanation = explainer.generate_explanation(processed_image.unsqueeze(0))
        
        result = {
            'predictions': predictions.numpy().tolist(),
            'explanation': explanation,
            'confidence': torch.max(torch.softmax(predictions, dim=1)).item()
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@bp.route('/batch_predict', methods=['POST'])
def batch_predict():
    files = request.files.getlist('files')
    results = []
    
    for file in files:
        try:
            image = Image.open(file.stream).convert('RGB')
            processed_image = preprocessor.process(image)
            
            with torch.no_grad():
                predictions = model(processed_image.unsqueeze(0))
            
            results.append({
                'filename': file.filename,
                'predictions': predictions.numpy().tolist(),
                'confidence': torch.max(torch.softmax(predictions, dim=1)).item()
            })
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e)
            })
    
    return jsonify({'results': results})

load_model()