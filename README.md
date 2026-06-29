# MedLink — Medical AI System

## Requirements
- Python 3.9+
- [Ollama](https://ollama.com) installed and running locally

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/Badrmohamed4/MedicalAISystem.git
cd MedicalAISystem

# 2. Create virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# Mac M1/M2/M3 ONLY — replace tensorflow line:
# pip uninstall tensorflow
# pip install tensorflow-macos tensorflow-metal

# 4. Pull Ollama model
ollama pull llama3

# 5. Run the app (from the MedicalAISystem folder)
python -m medical_chatbot.web_app
```

## Open in browser
```
http://127.0.0.1:5001
```

## Notes
- Ollama must be running before starting the app
- All 3 AI models (Brain, Skin, Lung) load automatically on first image upload
- Models are located in: `brainTumor case/`, `skinDisease case/`, `lungCancer case/`
