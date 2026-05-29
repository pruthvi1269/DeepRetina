# DeepRetina / Retinexa

# AI-Powered Diabetic Retinopathy Detection & Retinal Analysis System

DeepRetina (Retinexa) is an advanced AI-based healthcare web application developed for automated detection and classification of Diabetic Retinopathy (DR) from retinal fundus images using Deep Learning and Computer Vision techniques.

The system combines:
- Artificial Intelligence
- Medical Image Processing
- Explainable AI
- Web Technologies
- Deep Learning-based Diagnosis

to assist in early retinal disease screening and analysis.

---

# Key Features

## Secure User Authentication
- User Registration
- Login System
- Password Recovery via Email
- Session Management

---

# AI-Based Retina Validation
Before prediction, the system validates whether the uploaded image is a valid retinal fundus image using a Retina Validation CNN model.

Validation prevents:
- Invalid uploads
- Non-retinal images
- Corrupted inputs

---

# AI Disease Prediction

The AI model predicts 5 stages of Diabetic Retinopathy:

| Class | Prediction |
|------|------|
| 0 | No DR |
| 1 | Mild DR |
| 2 | Moderate DR |
| 3 | Severe DR |
| 4 | Proliferative DR |

---

# Explainable AI (Grad-CAM)

The system integrates Grad-CAM (Gradient-weighted Class Activation Mapping) for Explainable AI visualization.

Grad-CAM:
- Highlights infected retinal regions
- Shows AI attention areas
- Improves prediction transparency
- Assists medical interpretation

---

# Dynamic AI Clinical Analysis

The Analyse Dashboard dynamically generates:

- Diagnosis Prediction
- Risk Level
- Clinical Priority
- Treatment Recommendations
- Retinal Vessel Analysis
- Macular Edema Risk
- Optic Disc Analysis
- Glaucoma Monitoring Indicators

---

# AI Model Architecture

## Primary Deep Learning Model
- DenseNet121
- Transfer Learning
- ImageNet Pretrained Weights

## Training Techniques
- Mixed Precision Training
- Data Augmentation
- Fine-Tuning
- Class Weight Balancing
- ReduceLROnPlateau
- Early Stopping

---

# Algorithms & Techniques Used

## Deep Learning
- Convolutional Neural Networks (CNN)
- Transfer Learning
- DenseNet121 Architecture

## Computer Vision
- Image Preprocessing
- Retinal Image Normalization
- Image Rescaling
- Brightness Enhancement
- Data Augmentation

## Explainable AI
- Grad-CAM Heatmap Visualization

## Optimization Algorithms
- Adam Optimizer
- Categorical Crossentropy Loss

---

# Technologies Used

# Frontend
- HTML5
- CSS3
- JavaScript
- Bootstrap
- Glassmorphism UI
- 3D Animated Dashboard

---

# Backend
- Python
- Flask

---

# Database
- MySQL

---

# Artificial Intelligence & Libraries
- TensorFlow
- Keras
- OpenCV
- NumPy
- Scikit-learn
- Pillow

---

# Project Structure

```text
Retinexa/
│
├── app.py
├── train_dr_model.py
├── train_validation_model.py
├── retina_validator.h5
├── dr_weights.weights.h5
│
├── dataset/
│   ├── train/
│   └── val/
│
├── static/
│   ├── uploads/
│   ├── css/
│   ├── js/
│   ├── images/
│   └── heatmaps/
│
├── templates/
│   ├── home.html
│   ├── analyse.html
│   ├── login.html
│   ├── register.html
│   ├── forgot_password.html
│   └── about.html
│
├── README.md
└── requirements.txt