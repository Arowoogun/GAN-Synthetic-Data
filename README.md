# GAN-Synthetic-Data
This project develops a Generative Adversarial Network (GAN) model to generate synthetic geotechnical data that reproduces the statistical properties and relationships in the original dataset.  The model is implemented in Python and is based on the GAN framework introduced by Goodfellow et al. (2014).  
# GAN-Based Synthetic Geotechnical Data Generation

## Overview

This project develops a Generative Adversarial Network (GAN) to generate synthetic geotechnical data that reproduces the statistical characteristics and relationships present in an original dataset.
The model is implemented in Python using PyTorch and follows the Generative Adversarial Network framework introduced by Goodfellow et al. (2014).

The objective is to generate realistic synthetic observations for three geotechnical variables:

* **Depth** – Depth
* **Res** – Resistivity
* **Su_kpa** – Undrained shear strength

The synthetic dataset can potentially support geotechnical data augmentation and machine-learning applications where available field observations are limited.

## Dataset

The GAN was trained using a training dataset containing 301 observations and three variables:

| Variable | Description                    |
| -------- | ------------------------------ |
| Depth    | Depth              |
| Res      | Resistivity         |
| Su_kpa   | Undrained shear strength (kPa) |

Before GAN training, the variables were normalized using `MinMaxScaler`.

## GAN Architecture

The GAN consists of two neural networks:

### Generator

The generator transforms a 10-dimensional random latent vector into three synthetic geotechnical variables.

Architecture:

```text
Latent Vector (10)
      ↓
Dense Layer (128)
Batch Normalization
LeakyReLU
      ↓
Dense Layer (128)
Batch Normalization
LeakyReLU
      ↓
Dense Layer (64)
Batch Normalization
LeakyReLU
      ↓
Dense Layer (3)
Sigmoid
      ↓
Synthetic Geotechnical Data
```

### Discriminator

The discriminator receives the three geotechnical variables and attempts to distinguish real observations from GAN-generated observations.

```text
Input (3 variables)
      ↓
Dense Layer (128)
LeakyReLU + Dropout
      ↓
Dense Layer (128)
LeakyReLU + Dropout
      ↓
Dense Layer (64)
LeakyReLU + Dropout
      ↓
Output (Real/Fake)
```

## Model Training

The GAN was trained using:

* PyTorch
* Adam optimizer
* Binary Cross-Entropy with Logits Loss
* Learning rate: `0.0002`
* Batch size: `64`
* Latent dimension: `10`
* Training epochs: `4,000`
* Label smoothing for real observations
* Learning-rate scheduling

Random seeds were also specified to improve reproducibility.

## Synthetic Data Generation

After training, the generator was used to create:

**20,000 synthetic observations**

The normalized GAN outputs were transformed back to the original scale using the fitted `MinMaxScaler`.

The resulting synthetic dataset contains:

```text
Depth
Res
Su_kpa
```

## Model Evaluation

The statistical similarity between the original and synthetic datasets was evaluated using the two-sample **Kolmogorov–Smirnov (KS) test**.
The analysis compares the marginal distributions of:
* Depth
* Res
* Su_kpa

Additional visual evaluation is performed using:

* Histograms
* Scatter plots
* Kernel Density Estimation (KDE)
* Pairwise variable relationships

These analyses help evaluate whether the generated data reproduce important statistical patterns in the original dataset.

## Technologies

* Python
* PyTorch
* NumPy
* Pandas
* Scikit-learn
* SciPy
* Matplotlib
* Seaborn

## Repository Structure

```text
GAN-Synthetic-Geotechnical-Data/
│
├── README.md
├── GAN_Synthetic_Geotechnical_Data.ipynb
├── data/
├── outputs/
├── requirements.txt
└── .gitignore
```

##  Application
Synthetic geotechnical data created using this code were used to augment geotechnical data training and subsequent machine learning prediction of Su in Arowoogun et al., 2025



