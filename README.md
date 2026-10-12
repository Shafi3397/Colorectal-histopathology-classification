# Colorectal Histopathology Classification Using Deep Learning

**An MSc Deep Learning Course Project | PyTorch | Medical Image Classification**

## Overview

This project explores automated classification of colorectal cancer histopathology images using deep learning. It implements **EfficientNet-B0** with transfer learning and compares its performance against a **ResNet-50** baseline.

The study investigates model optimization, generalization, training dynamics, and classification performance across nine colorectal tissue categories.

The project was developed as part of an MSc Deep Learning course assignment.

## Dataset

**Dataset:** NCT-CRC-HE-100K

The dataset contains 100,000 non-overlapping histological image patches representing nine tissue classes.

| Class | Tissue Type |
|---|---|
| ADI | Adipose tissue |
| BACK | Background |
| DEB | Debris |
| LYM | Lymphocytes |
| MUC | Mucus |
| MUS | Smooth muscle |
| NORM | Normal colon mucosa |
| STR | Cancer-associated stroma |
| TUM | Colorectal adenocarcinoma epithelium |

**Dataset source:** [NCT-CRC-HE-100K on Kaggle](https://www.kaggle.com/datasets/imrankhan77/nct-crc-he-100k)

### Data Partitioning

The implemented pipeline uses a reproducible random image-level split:

| Subset | Images | Proportion |
|---|---:|---:|
| Training | 72,000 | 72% |
| Validation | 18,000 | 18% |
| Testing | 10,000 | 10% |

**Important:** The split is not explicitly stratified or patient-/slide-independent. Consequently, the reported performance should be interpreted as internal holdout performance rather than independent clinical generalization.

## Methodology

### 1. Image Preprocessing and Augmentation

Images are resized or cropped to 224 × 224 pixels and normalized using ImageNet statistics.

Training augmentation includes:
- Random resized cropping
- Horizontal and vertical flipping
- Random rotation
- Color jittering
- Gaussian blurring

### 2. Model Architectures

**EfficientNet-B0**

A pretrained EfficientNet-B0 backbone is adapted for nine-class tissue classification through a custom classification head.

Key architectural features include MBConv blocks, squeeze-and-excitation mechanisms, and efficient model scaling.

**ResNet-50**

A pretrained ResNet-50 model is implemented as a comparative baseline. Its residual connections facilitate gradient propagation in deeper networks.

### 3. Transfer Learning and Training

Both architectures use a two-phase training strategy:

**Phase 1: Classifier Head Training**
- Pretrained backbone frozen
- Maximum 5 epochs
- Learning rate: 0.001

**Phase 2: Full Fine-Tuning**
- All layers unfrozen
- Maximum 25 epochs
- Learning rate: 0.0001

Training configuration:

| Parameter | Configuration |
|---|---|
| Optimizer | AdamW |
| Loss Function | Cross-Entropy |
| Batch Size | 64 |
| Weight Decay | 0.00001 |
| Dropout | 0.3 |
| Learning Rate Scheduler | Cosine Annealing |
| Early Stopping Patience | 5 |
| Gradient Clipping | Maximum norm of 1.0 |
| Random Seed | 42 |

### 4. Performance Evaluation

The evaluation pipeline computes classification accuracy, macro F1-score, Cohen's Kappa, macro AUC-ROC, class-wise precision and recall, and confusion matrices.

Additional analysis includes:
- Training and validation learning curves
- Learning rate visualization
- Gradient monitoring
- Comparative evaluation against ResNet-50
- Grad-CAM implementation for model interpretability

## Experimental Results

The original experiment reported the following EfficientNet-B0 results on its 10,000-image internal test partition:

| Evaluation Metric | EfficientNet-B0 |
|---|---:|
| Test Accuracy | 99.74% |
| Macro F1-Score | 99.73% |
| Cohen's Kappa | 0.9971 |
| Macro AUC-ROC | 1.0000 |
| Test Loss | 0.0095 |

These values were recorded in the original experiment logs and have not been independently reproduced for this repository.

The implementation also includes a ResNet-50 comparison pipeline.

## Explainable AI and Gradient Analysis

**Gradient Monitoring:** The project examines layer-wise gradient magnitudes to investigate gradient propagation during training.

**Grad-CAM:** The code includes a Gradient-weighted Class Activation Mapping implementation intended to visualize image regions influencing predictions.

The Grad-CAM execution was not completed in the supplied experiment log because its dependency was unavailable. Successful Grad-CAM visualizations are therefore not claimed as completed results.

## Technologies Used

- Python
- PyTorch and Torchvision
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Grad-CAM (optional)
- Kaggle GPU environment

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/Shafi3397/colorectal-histopathology-classification.git
cd colorectal-histopathology-classification
```

### 2. Install dependencies

```bash
pip install torch torchvision numpy scikit-learn matplotlib seaborn grad-cam
```

### 3. Download the dataset

Download NCT-CRC-HE-100K from the dataset link above.

The provided source code is configured for the original Kaggle dataset path. Update `DATA_DIR` and `OUTPUT_DIR` in the Python script when running in another environment.

### 4. Run the experiment

```bash
python histopathology_classification.py
```

Training both models on the full dataset is computationally intensive. A CUDA-enabled GPU is recommended.

## Limitations

- Random image-level partitioning does not guarantee patient- or slide-independent evaluation.
- Independent external validation was not performed in the documented experiment.
- Grad-CAM visualizations were not successfully generated in the available execution log.
- The reported results are associated with the original experimental configuration and should not be interpreted as clinical diagnostic validation.

## Future Improvements

Potential extensions include:
- Patient- or slide-independent evaluation protocols
- External validation using independent histopathology datasets
- Stain normalization and robustness analysis
- Improved explainability evaluation
- Additional architecture comparisons

## Academic Context

This repository documents an MSc-level Deep Learning coursework project focused on medical image classification, transfer learning, model benchmarking, and experimental evaluation.

**Project Type:** Academic Coursework  
**Field:** Deep Learning / Computer Vision / Medical Image Analysis  
**Framework:** PyTorch

## Disclaimer

This project is intended for academic and educational purposes only. It is not a clinically validated diagnostic system.
