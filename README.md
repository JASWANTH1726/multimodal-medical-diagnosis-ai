# Multimodal Medical Diagnosis AI

Advanced multimodal medical diagnosis system with explainable AI that combines medical imaging and clinical text data to provide interpretable diagnostic predictions.

## Quick Start

```bash
git clone https://github.com/JASWANTH1726/multimodal-medical-diagnosis-ai.git
cd multimodal-medical-diagnosis-ai
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python api/app.py
```

## Project Structure

- `data/` - Data loading and preprocessing
- `models/` - Model architectures (encoders, fusion, classification)
- `training/` - Training pipeline and evaluation
- `explainability/` - Model interpretation tools
- `api/` - FastAPI server for predictions
- `tests/` - Unit and integration tests
- `notebooks/` - Jupyter notebooks for exploration
- `docs/` - Architecture and deployment documentation

## Features

- **Multimodal Learning**: Combines medical images and clinical text
- **Explainability**: Attention maps, feature importance, Grad-CAM
- **Easy Deployment**: Docker support included
- **API Server**: FastAPI with comprehensive endpoints

## Requirements

- Python 3.9+
- CUDA 11.8+ (optional, for GPU)
- See requirements.txt for dependencies

## License

MIT License
