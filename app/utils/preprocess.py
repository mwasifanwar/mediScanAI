import torch
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
import numpy as np
import cv2

class MedicalImagePreprocessor:
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def process(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        return self.transform(image)
    
    def enhance_contrast(self, image, factor=2.0):
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    def remove_noise(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        image = cv2.medianBlur(image, 3)
        return Image.fromarray(image)
    
    def adjust_gamma(self, image, gamma=1.2):
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        
        image = cv2.LUT(image, table)
        return Image.fromarray(image)

class AdvancedPreprocessor:
    def __init__(self):
        self.preprocessor = MedicalImagePreprocessor()
    
    def process_for_training(self, image):
        image = self.preprocessor.enhance_contrast(image)
        image = self.preprocessor.remove_noise(image)
        image = self.preprocessor.adjust_gamma(image)
        return self.preprocessor.process(image)
    
    def process_for_inference(self, image):
        return self.preprocessor.process(image)

class DICOMProcessor:
    def __init__(self):
        pass
    
    def load_dicom(self, dicom_path):
        try:
            import pydicom
            ds = pydicom.dcmread(dicom_path)
            image = ds.pixel_array
            image = self._normalize_dicom(image)
            return Image.fromarray(image)
        except ImportError:
            raise ImportError("pydicom is required for DICOM processing")
    
    def _normalize_dicom(self, image):
        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min()) * 255.0
        return image.astype(np.uint8)