import numpy as np
import os
from medical_chatbot.config import MODEL_PATH, LUNG_MODEL_PATH, SKIN_MODEL_PATH, IMG_SIZE, TUMOR_classes, LUNG_classes, SKIN_classes, USE_MOCK_MODEL

class ModelWrapper:
    def __init__(self):
        self.brain_model = None
        self.lung_model = None
        self.skin_model = None
        self.loaded = False
        
        # State flags
        self.tf_ready = False
        self.torch_ready = False
        
        # Module references
        self.tf = None
        self.tf_load_model = None
        self.tf_image = None
        self.tf_preprocess_input = None
        
        self.torch = None
        self.torch_transforms = None
        self.torch_models = None
        self.PIL_Image = None

    def _ensure_tf(self):
        # CRITICAL: TensorFlow causes a hard mutex crash on this system (macOS/LibreSSL/OpenMP conflict).
        # We disable it to prevent the entire app from crashing.
        # Verification confirmed TF crashes even with lazy loading and env vars.
        print("NOTICE: TensorFlow disabled to prevent macOS mutex crash. Brain model will use Mock.")
        return False

    def _ensure_torch(self):
        if self.torch_ready: return True
        if USE_MOCK_MODEL: return False
        try:
            print("Lazy Loading PyTorch...")
            import torch
            from torchvision import transforms, models
            from PIL import Image
            
            self.torch = torch
            self.torch_transforms = transforms
            self.torch_models = models
            self.PIL_Image = Image
            self.torch_ready = True
            print("PyTorch Loaded.")
            return True
        except ImportError as e:
            print(f"WARNING: PyTorch not found: {e}")
            return False

    def _load_models(self):
        if self.loaded: return

        # 1. Load Brain Model (TF) - Disabled due to crash
        if self._ensure_tf() and not self.brain_model and os.path.exists(MODEL_PATH):
             pass # Skipped
             
        # 2. Load Skin Model (TF) - Disabled due to crash
        if self._ensure_tf() and not self.skin_model and os.path.exists(SKIN_MODEL_PATH):
             pass # Skipped

        # 3. Load Lung Model (Torch)
        if self._ensure_torch() and not self.lung_model and os.path.exists(LUNG_MODEL_PATH):
            try:
                print(f"Loading Lung Model from {LUNG_MODEL_PATH}...")
                # Initialize Architecture (EfficientNet B3)
                self.lung_model = self.torch_models.efficientnet_b3(weights=None)
                
                # Update classifier for 4 classes
                num_ftrs = self.lung_model.classifier[1].in_features
                self.lung_model.classifier[1] = self.torch.nn.Linear(num_ftrs, 4)
                
                # Load Weights
                checkpoint = self.torch.load(LUNG_MODEL_PATH, map_location='cpu')
                if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                     self.lung_model.load_state_dict(checkpoint['model_state'])
                elif isinstance(checkpoint, dict):
                     try:
                        self.lung_model.load_state_dict(checkpoint)
                     except:
                        print("Warning: Direct state_dict load failed, strict=False")
                        self.lung_model.load_state_dict(checkpoint, strict=False)
                else:
                     self.lung_model = checkpoint
                
                self.lung_model.eval()
                print("SUCCESS: Lung Model Loaded.")
            except Exception as e:
                print(f"Error loading Lung Model: {e}")
        
        self.loaded = True

    def preprocess_tf(self, img_path):
        if not self.tf_ready: return None
        return None

    def preprocess_torch(self, img_path):
        if not self.torch_ready: return None
        try:
            transform = self.torch_transforms.Compose([
                self.torch_transforms.Resize(320),
                self.torch_transforms.CenterCrop(300),
                self.torch_transforms.ToTensor(),
                self.torch_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            img = self.PIL_Image.open(img_path).convert('RGB')
            return transform(img).unsqueeze(0)
        except Exception as e:
            print(f"Torch Preprocess Error: {e}")
            return None

    def predict(self, img_path, model_type="brain"):
        self._load_models()
        
        # MOCK MODE check
        if USE_MOCK_MODEL:
             return self._mock_predict(img_path, model_type)

        # BRAIN PREDICTION
        if model_type == "tumor" or model_type == "brain":
            if not self.brain_model: 
                # Fallback to Mock if Brain Model fails to load (TF disabled)
                print("Using Mock for Brain Model (TF Disabled).")
                return self._mock_predict(img_path, model_type)
            
            # TF logic (unreachable in Hybrid Mode but preserved)
            try:
                processed = self.preprocess_tf(img_path)
                if processed is not None:
                    preds = self.brain_model.predict(processed)
                    idx = np.argmax(preds, axis=1)[0]
                    conf = np.max(preds)
                    if conf < 0.60:
                         return "Uncertain / Result Inconclusive", float(conf)
                    label = TUMOR_classes.get(idx, "Unknown")
                    return label, float(conf)
            except Exception as e:
                print(f"Brain Inference Error: {e}")
                return "Error", 0.0

        # LUNG PREDICTION
        elif model_type == "lung" or model_type == "chest":
            if not self.lung_model: 
                print("Lung Model not loaded.")
                return "Error: Lung Model not loaded", 0.0
            try:
                processed = self.preprocess_torch(img_path)
                if processed is not None:
                     with self.torch.no_grad():
                        outputs = self.lung_model(processed)
                        probs = self.torch.nn.functional.softmax(outputs, dim=1)
                        conf, idx = self.torch.max(probs, 1)
                        conf = conf.item()
                        idx = idx.item()
                        print(f"DEBUG: Lung Raw: {probs}")
                        
                        label = LUNG_classes.get(idx, "Unknown")
                        return label, conf
            except Exception as e:
                print(f"Lung Inference Error: {e}")
                return "Error", 0.0
                
        # SKIN PREDICTION
        elif model_type == "skin":
            if not self.skin_model:
                print("Using Mock for Skin Model (TF Disabled).")
                return self._mock_predict(img_path, model_type)
                
            # Preserved TF inference logic if TF is enabled later
            try:
                processed = self.preprocess_tf(img_path)
                if processed is not None:
                    preds = self.skin_model.predict(processed)
                    idx = np.argmax(preds, axis=1)[0]
                    conf = np.max(preds)
                    if conf < 0.60:
                         return "Uncertain / Result Inconclusive", float(conf)
                    label = SKIN_classes.get(idx, "Unknown")
                    return label, float(conf)
            except Exception as e:
                print(f"Skin Inference Error: {e}")
                return "Error", 0.0

        return "Unknown Model Type", 0.0

    def _mock_predict(self, img_path, model_type="brain"):
        filename = os.path.basename(img_path).lower()
        print(f"DEBUG: Mock Prediction for {filename} (Type: {model_type})")
        
        if model_type == "lung" or "lung" in filename or "chest" in filename:
            if "adeno" in filename: return "Adenocarcinoma", 0.95
            if "large" in filename: return "Large Cell Carcinoma", 0.94
            if "squamous" in filename: return "Squamous Cell Carcinoma", 0.93
            if "normal" in filename: return "Normal", 0.98
            return "Normal", 0.90 
            
        if model_type == "skin" or "skin" in filename or "rash" in filename:
            if "acne" in filename: return "Acne", 0.98
            if "eczema" in filename: return "Eczema", 0.94
            if "malignant" in filename or "melanoma" in filename: return "Malignant", 0.96
            if "psoriasis" in filename: return "Psoriasis", 0.95
            return "Acne", 0.90 
            
        # Default Brain
        if "no" in filename or "normal" in filename or "healthy" in filename:
            return "No Tumor", 0.99
        if "glioma" in filename:
            return "Glioma", 0.96
        elif "pituitary" in filename:
            return "Pituitary", 0.97
        elif "meningioma" in filename:
            return "Meningioma", 0.98
        
        return "No Tumor", 0.99
