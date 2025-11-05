import pandas as pd
import numpy as np
from PIL import Image
import os

def create_sample_dataset(output_dir='data/sample_images', csv_path='data/sample_data.csv'):
    os.makedirs(output_dir, exist_ok=True)
    
    sample_data = []
    diseases = ['normal', 'pneumonia', 'covid', 'fracture', 'cancer']
    
    for i in range(100):
        disease = np.random.choice(diseases)
        
        image = create_sample_image(disease)
        filename = f"sample_{i:03d}_{disease}.png"
        image_path = os.path.join(output_dir, filename)
        image.save(image_path)
        
        sample_data.append({
            'image_filename': filename,
            'disease': disease,
            'severity': np.random.choice(['mild', 'moderate', 'severe']),
            'age': np.random.randint(20, 80),
            'gender': np.random.randint(0, 2),
            'temperature': np.random.uniform(36.0, 39.5),
            'bp_systolic': np.random.randint(100, 180),
            'bp_diastolic': np.random.randint(60, 110),
            'heart_rate': np.random.randint(60, 120),
            'respiratory_rate': np.random.randint(12, 30),
            'oxygen_saturation': np.random.uniform(90.0, 100.0),
            'white_blood_cells': np.random.uniform(4.0, 15.0),
            'c_reactive_protein': np.random.uniform(0.0, 10.0)
        })
    
    df = pd.DataFrame(sample_data)
    df.to_csv(csv_path, index=False)
    print(f"Sample dataset created with {len(sample_data)} images")

def create_sample_image(disease):
    width, height = 224, 224
    image = Image.new('RGB', (width, height), color='white')
    
    return image