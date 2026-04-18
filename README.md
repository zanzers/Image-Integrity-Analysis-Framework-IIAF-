# Image Integrity Analysis Framework (IIAF)

A machine learning-based framework designed to analyze images and detect whether they are **original** or **AI-tampered**, including classification of tampering types such as face swap, deepfake, and AI-generated content insertion.

---

## 📌 Overview

The **Image Integrity Analysis Framework (IIAF)** performs:

* Image authenticity verification
* Detection of AI-based tampering
* Classification of tampering types
* Feature-based and CNN-assisted analysis

---

## 🧠 Supported Classes

| Label | Description          |
| ----- | -------------------- |
| 0     | Original             |
| 1     | Face Swap            |
| 2     | AI Content Insertion |
| 3     | Deepfake             |

---

## 📁 Expected Folder Structure

```
Cnn_datasets/
├── original/
├── faceswap/
├── ai_content_insertion/
└── deepfake/
```

> ⚠️ Make sure folder names match your label mapping during training.

---

---

## 🚀 How It Works

1. Input image is preprocessed
2. Features are extracted (ELA, PRNU, IWT)
3. CNN (lightweight) extracts deep features
4. Classifier predicts tampering type
5. Result is returned with confidence score

---

## 🛠️ Requirements

* Python 3.x
* OpenCV
* NumPy
* PyTorch
* scikit-learn
* openpyxl (optional for reporting)

---

## 📅 Development Tasks

### ✅ Today

* [ ] Setup project structure
* [ ] Configure virtual environment
* [ ] Install dependencies
* [ ] Prepare dataset folders
* [ ] Implement basic training pipeline (`train.py`)

---

### 🔜 Tomorrow

* [ ] Integrate feature extraction (ELA, PRNU, IWT)
* [ ] Replace placeholder features with real pipeline
* [ ] Train initial model
* [ ] Test single image inference (`main.py`)
* [ ] Optimize model performance

---

## 💡 Notes

* Ensure dataset is balanced across all classes
* Keep model lightweight for deployment
* Avoid large model files when pushing to Git

---

## 📖 Thesis Context

This project follows a **client-server architecture**, where computationally intensive image analysis is performed on a backend server, enabling scalable and efficient processing.

---
