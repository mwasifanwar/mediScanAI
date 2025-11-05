import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self._register_hooks()
    
    def _register_hooks(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        def forward_hook(module, input, output):
            self.activations = output
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor, target_class=None):
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        
        output.backward(gradient=one_hot)
        
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_tensor.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        
        return cam

class ExplainableAI:
    def __init__(self, model):
        self.model = model
        self.target_layer = self._find_target_layer()
        self.grad_cam = GradCAM(model, self.target_layer)
    
    def _find_target_layer(self):
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                return module
        raise ValueError("No convolutional layer found")
    
    def generate_explanation(self, input_tensor, target_class=None):
        cam = self.grad_cam.generate(input_tensor, target_class)
        
        input_image = input_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
        
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[..., ::-1]
        
        superimposed = heatmap + np.float32(input_image)
        superimposed = superimposed / np.max(superimposed)
        
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(input_image)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(cam, cmap='jet')
        plt.title('Attention Map')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(superimposed)
        plt.title('Overlay')
        plt.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return image_base64
    
    def generate_confidence_map(self, input_tensor):
        with torch.no_grad():
            features = self.model.image_encoder(input_tensor)
            probabilities = F.softmax(self.model.disease_head(features), dim=1)
        
        return probabilities.cpu().numpy()