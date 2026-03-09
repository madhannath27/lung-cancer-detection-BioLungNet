# 🧬 BioLungNet: AI-Powered Lung Cancer Detection from CT Scan Images

## Overview

BioLungNet is a deep learning-based framework designed for automated lung cancer detection using CT scan images.

The model utilizes **DenseNet121 as the backbone architecture** combined with custom classification layers to classify CT images into three categories:

- Normal
- Benign
- Malignant

The project applies **transfer learning, data augmentation, and deep convolutional neural networks** to improve diagnostic accuracy for computer-aided medical diagnosis systems.

---

## Author

**Madhannath S**  
B.Tech Biotechnology  
Department of Biotechnology  
K.S. Rangasamy College of Technology  
Tiruchengode, Tamil Nadu, India

---

## Key Features

- Deep learning model using DenseNet121
- Transfer learning for improved feature extraction
- Albumentations-based data augmentation
- Multi-class classification (Normal / Benign / Malignant)
- Grad-CAM model interpretability
- Single image prediction support
- Confusion matrix and ROC curve evaluation

---

## Dataset

The model is trained using CT scan images categorized into three diagnostic classes.

### Dataset Classes

- Benign
- Malignant
- Normal

### Dataset Description

**Original Dataset**

Contains raw CT scan images collected from the dataset source.

**Trained Dataset**

Contains processed and augmented images used for model training.

---

## Data Augmentation

Due to the limited dataset size, the **Albumentations library** was used to increase dataset diversity.

### Augmentation Techniques Used

- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Random Brightness and Contrast
- Gamma Correction
- Gaussian Noise
- Affine Transformations
- Elastic Transformations
- Horizontal Flip
- Coarse Dropout

All images were resized to:

224 × 224 pixels

before training.

---

## Model Architecture

### Backbone Network

DenseNet121 (Pretrained on ImageNet)

### Custom Classification Head

```python
GlobalAveragePooling2D
Dense(512)
LeakyReLU
BatchNormalization
Dropout(0.5)

Dense(256)
LeakyReLU
BatchNormalization
Dropout(0.3)

Dense(3) Softmax
```

This architecture enables efficient feature reuse and improves classification accuracy.

---

## Training Strategy

### Phase 1 – Feature Extraction

DenseNet backbone frozen while training custom layers.

### Phase 2 – Fine Tuning

Upper DenseNet layers unfrozen for improved feature learning.

### Callbacks Used

- EarlyStopping
- ReduceLROnPlateau
- ModelCheckpoint

---

## Evaluation Metrics

The model performance was evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve
- AUC Score
- Confusion Matrix

---

## Results

### Overall Performance

| Metric | Score |
|------|------|
| Accuracy | 94.77% |
| Weighted Recall | 0.9477 |
| Weighted F1 Score | 0.9477 |

### Per-Class Performance

| Class | Precision | Recall | F1 Score | AUC |
|------|------|------|------|------|
| Benign | 0.91 | 0.91 | 0.91 | 0.979 |
| Malignant | 0.97 | 0.98 | 0.97 | 1.000 |
| Normal | 0.95 | 0.96 | 0.95 | 0.983 |

---

## Model Interpretability

Grad-CAM visualization was used to understand model predictions.

The heatmaps highlight lung regions that influenced the classification results, helping validate model decisions.

---

## Project Structure

```
BioLungNet-Lung-Cancer-Detection/
│
├── Dataset/
│   ├── Original Dataset/
│   │   ├── Benign cases/
│   │   ├── Malignant cases/
│   │   └── Normal cases/
│   │
│   └── Trained Dataset/
│       ├── Benign/
│       ├── Malignant/
│       └── Normal/
│
├── Program/
│   ├── Model/
│   │   └── Saved trained model files
│   │
│   ├── AugmentationProgram.py
│   │
│   ├── BioLungNet_Main.ipynb
│   │
│   └── Single_Image_Predict.py
│
├── Result Images/
│   ├── confusion_matrix.png
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   └── gradcam_visualization.png
```

---

## Prediction Output

The script will classify the CT scan image as:

- Normal
- Benign
- Malignant

---

## Technologies Used

- Python
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- Albumentations
- Jupyter Notebook

---

## Future Improvements

- Integration with 3D CT scan models
- Hybrid CNN–Transformer architecture
- Larger clinical datasets
- Real-time clinical deployment

---

## License

This project is licensed under the **MIT License**.

---

## Keywords

Lung Cancer Detection, Deep Learning, CT Scan Analysis, DenseNet121, Transfer Learning, BioLungNet, Medical Image Processing.
