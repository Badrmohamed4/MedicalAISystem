import os
import numpy as np
from PIL import Image

# Prevent TF messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class ImagePipeline:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        
        # Lazy loading placeholders
        self.brain_model = None
        self.lung_model = None
        self.skin_model = None

    def _load_brain_model(self):
        """
        Manual h5 loading to prevent Keras version conflicts.
        """
        if self.brain_model is not None:
            return
            
        import h5py
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
        from tensorflow.keras.applications import EfficientNetB0
        
        print("Loading Brain Model...")
        model_path = os.path.join(self.models_dir, 'brain_tumor_classifier.h5')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model at {model_path}")

        base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
        self.brain_model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            BatchNormalization(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(4, activation='softmax')
        ])
        
        # Load weights
        self.brain_model.load_weights(model_path)
        print("Brain Model loaded successfully.")

    def _load_lung_model(self):
        """
        PyTorch EfficientNetB3 loading.
        """
        if self.lung_model is not None:
            return
            
        import torch
        import torchvision.models as models
        
        print("Loading Lung Model...")
        model_path = os.path.join(self.models_dir, 'lung_cancer_efficientnet_b3_final3.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model at {model_path}")

        # Initialize base model architecture
        self.lung_model = models.efficientnet_b3(weights=None)
        
        # Replace classifier head for 4 classes
        num_ftrs = self.lung_model.classifier[1].in_features
        self.lung_model.classifier[1] = torch.nn.Linear(num_ftrs, 4)
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        self.lung_model.load_state_dict(state_dict)
        self.lung_model.eval()
        print("Lung Model loaded successfully.")

    def _load_skin_model(self):
        """
        Skin model loading from model.weights.h5
        """
        if self.skin_model is not None:
            return
            
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
        from tensorflow.keras.applications import MobileNetV2
        
        print("Loading Skin Model...")
        model_path = os.path.join(self.models_dir, 'model.weights.h5')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model at {model_path}")

        # Assume 7 classes for skin lesions, adjust if different
        base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
        self.skin_model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(7, activation='softmax')
        ])
        
        self.skin_model.load_weights(model_path)
        print("Skin Model loaded successfully.")

    def predict(self, image_path, domain):
        try:
            image = Image.open(image_path).convert('RGB')
            image = image.resize((224, 224))
            
            if domain == "brain":
                self._load_brain_model()
                from tensorflow.keras.applications.efficientnet import preprocess_input
                img_array = np.array(image)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)
                
                preds = self.brain_model.predict(img_array, verbose=0)
                classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
                idx = np.argmax(preds[0])
                conf = preds[0][idx] * 100
                return f"Domain analysis indicates: {classes[idx]} (Confidence {conf:.2f}%)"
                
            elif domain == "lung":
                self._load_lung_model()
                import torch
                from torchvision import transforms
                
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                
                img_tensor = transform(image).unsqueeze(0)
                with torch.no_grad():
                    outputs = self.lung_model(img_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    conf, idx = torch.max(probs, 1)
                    
                classes = ["adenocarcinoma", "large.cell.carcinoma", "normal", "squamous.cell.carcinoma"]
                return f"Domain analysis indicates: {classes[idx.item()]} (Confidence {conf.item()*100:.2f}%)"
                
            elif domain == "skin":
                self._load_skin_model()
                # Assuming MobileNetV2 preprocessing (normalize -1 to 1)
                img_array = np.array(image, dtype=np.float32)
                img_array = (img_array / 127.5) - 1.0
                img_array = np.expand_dims(img_array, axis=0)
                
                preds = self.skin_model.predict(img_array, verbose=0)
                # Generic classes for demonstration, update with actual skin classes if needed
                classes = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Class 6"]
                idx = np.argmax(preds[0])
                conf = preds[0][idx] * 100
                return f"Domain analysis indicates: {classes[idx]} (Confidence {conf:.2f}%)"
                
            else:
                return "Unknown domain for image analysis."
                
        except Exception as e:
            return f"Error during image analysis: {str(e)}"
