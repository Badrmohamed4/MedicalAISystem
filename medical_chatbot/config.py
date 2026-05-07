import os

# Base Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model Paths
# Using the absolute path found in the user's workspace
# Model Paths
# Using the absolute path found in the user's workspace
MODEL_PATH = "/Users/badrabdelrazek/Desktop/UNI/GradProject/chatbot/brainTumor case/brain_tumor_classifier.h5"
LUNG_MODEL_PATH = "/Users/badrabdelrazek/Desktop/UNI/GradProject/chatbot/lungCancer case/lung_cancer_efficientnet_b3_final3.pth"
SKIN_MODEL_PATH = "/Users/badrabdelrazek/Desktop/UNI/GradProject/chatbot/skinDisease case/model.weights.h5"

# System Configuration
# Set to True if TensorFlow crashes on your system (e.g. Mac M1/M2 Mutex Lock)
USE_MOCK_MODEL = False


# NLP Configuration
INTENT_THRESHOLD = 0.6
URGENCY_KEYWORDS = ["pain", "blood", "seizure", "unconscious", "breeding", "severe", "worst"]

# Medical Knowledge Base (Simple Rule-based)
TUMOR_classes = {
    0: "Glioma",
    1: "Meningioma",
    2: "No Tumor",
    3: "Pituitary"
}

LUNG_classes = {
    0: "Adenocarcinoma",
    1: "Large Cell Carcinoma",
    2: "Normal",
    3: "Squamous Cell Carcinoma"
}

SKIN_classes = {
    0: "Acne",
    1: "Eczema",
    2: "Malignant",
    3: "Psoriasis"
}

# Image Configuration
IMG_SIZE = (224, 224) # Standard for many CNNs, verify if different