# LungAI - Lung Cancer Detection System

A deep learning-based system for automated lung cancer classification using histopathological images. This project uses a ResNet-50 architecture to classify lung tissue images into four categories: Adenocarcinoma, Large Cell Carcinoma, Normal, and Squamous Cell Carcinoma.

## 🎯 Overview

LungAI is a PyTorch-based medical image classification system that leverages transfer learning with ResNet-50 to detect and classify different types of lung cancer from histopathological images. This project represents the newest version, now using PyTorch for improved performance and flexibility.

## 📈 Model Performance

The model achieves impressive results:

- **98% accuracy** in distinguishing between cancerous and non-cancerous cases
- **83% overall accuracy** in differentiating between four specific types of lung conditions

### Per-Class Performance (F1-Scores)

| Cancer Type | F1-Score |
|------------|----------|
| **Adenocarcinoma** | 82% |
| **Large Cell Carcinoma** | 85% |
| **Normal (non-cancerous)** | 98% |
| **Squamous Cell Carcinoma** | 76% |

## 🔬 Cancer Types Detected

- **Adenocarcinoma**: The most common type of lung cancer
- **Large Cell Carcinoma**: An aggressive form of non-small cell lung cancer
- **Squamous Cell Carcinoma**: A type of non-small cell lung cancer
- **Normal**: Healthy lung tissue

## 🏗️ Architecture

The model uses a modified ResNet-50 architecture with:
- Pre-trained weights from ImageNet (ResNet50_Weights.IMAGENET1K_V1)
- Custom fully connected layers:
  - Linear layer (2048 → 256)
  - ReLU activation
  - Dropout (0.5)
  - Linear layer (256 → 4 classes)

## 📁 Repository Structure

```
LungAI-main/
│
├── app.py                     # Gradio web interface for predictions
├── run.py                     # Command-line inference script
├── architecture.py            # Model architecture and training code
├── preprocess.py             # Data preprocessing utilities
├── requirements.txt          # List of Python dependencies
├── README.md                 # Project documentation
│
├── Model/                    # Stores trained model files
│   ├── lung_cancer_detection_model.pth    # PyTorch weights
│   └── lung_cancer_detection_model.onnx   # ONNX format
│
├── Data/                     # (Not included) Dataset directory
├── Processed_Data/           # (Not included) Preprocessed data
└── assets/                   # Additional project assets
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd LungAI-main
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Web Interface (Gradio)

Launch the interactive web interface:

```bash
python app.py
```

This will start a Gradio interface where you can:
- Upload lung tissue images
- Get real-time predictions
- View example predictions

The interface will be available at `http://localhost:7860` and will also generate a public shareable link.

### Command-Line Inference

Run predictions on individual images:

```bash
python run.py
```

Edit the `image_path` variable in [run.py](run.py) to specify your own image.

## 📦 Dependencies

- **torch**: Deep learning framework
- **torchvision**: Computer vision utilities and models
- **Pillow**: Image processing
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning utilities
- **gradio**: Web interface framework
- **onnx & onnxruntime**: Model export and inference

## 🎓 Model Training

The model uses several techniques to improve performance:

### Data Augmentation
- Random resized crop (224x224)
- Random horizontal flip
- Random rotation (±10 degrees)
- Color jitter (brightness, contrast, saturation)

### Training Configuration
- Optimizer: Adam with learning rate scheduling
- Learning rate: Adaptive with ReduceLROnPlateau scheduler
- Dropout: 0.5 for regularization
- Image normalization: ImageNet statistics

### Preprocessing
All images are:
- Resized to 256x256
- Center-cropped to 224x224
- Normalized with ImageNet mean and std
- Converted to tensors

## 📊 Model Files

The repository includes two model formats:

1. **PyTorch (.pth)**: Native PyTorch format for training and inference
2. **ONNX (.onnx)**: Cross-platform format for deployment

## 🖼️ Input Requirements

- **Format**: PNG, JPG, or any standard image format
- **Type**: Histopathological lung tissue images
- **Processing**: Images are automatically resized and normalized

## 🔧 GPU Support

The system automatically detects and uses CUDA-enabled GPUs if available:
```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

CPU inference is supported but will be slower.

## 📝 Notes

- The model is trained on histopathological images and should only be used with similar medical imaging data
- This is a research/educational tool and should not replace professional medical diagnosis
- Ensure proper data privacy and ethical considerations when handling medical images

## 🙏 Acknowledgments

- ResNet-50 architecture from torchvision
- Pre-trained weights from ImageNet
- Gradio for the web interface framework

---

**Disclaimer**: This tool is for research and educational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.
