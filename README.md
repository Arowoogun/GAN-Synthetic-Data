# GAN-Based Synthetic Geotechnical Data Generation

## Overview
This repository contains a **Generative Adversarial Network (GAN)** developed in Python using **PyTorch** to generate synthetic geotechnical and geophysical data.
The GAN was developed to address limitations associated with sparse paired geotechnical and geophysical observations. It learns the joint characteristics of three variables:

* **Depth**
* **Electrical Resistivity (Res)**
* **Undrained Shear Strength (Su)**

The trained generator is used to produce synthetic observations that can augment the original dataset and support downstream machine-learning applications.
This work is associated with research on predicting undrained shear strength from **Towed Transient Electromagnetic (Towed-TEM) resistivity data** and sparse **Cone Penetration Test (CPT)** measurements.

---

## Associated Publication
The methodology implemented in this repository is associated with the following research article:

**Arowoogun, K., Grote, K., & Maurer, J. (2026).**
**Predicting undrained shear strength from Towed-TEM resistivity and sparse CPT data using machine learning models.**
*Journal of Applied Geophysics*, **254**, Article 106500.

**DOI:**
https://doi.org/10.1016/j.jappgeo.2026.106500
The article is available through the *Journal of Applied Geophysics* on ScienceDirect.

---

## Research Motivation

Geotechnical investigations often contain relatively sparse measurements because field sampling and Cone Penetration Testing can be costly and spatially limited. Geophysical methods such as Towed-TEM can provide much denser spatial coverage, but translating geophysical measurements into geo-engineering parameters such as undrained shear strength requires sufficient paired observations for model development.
Synthetic-data generation provides one approach for increasing the amount of training data available while attempting to preserve patterns contained in the observed dataset.

This project applies a GAN to learn relationships among:
```text
Depth
Electrical Resistivity
Undrained Shear Strength (Su)
```

and generate additional synthetic observations.

---

## GAN Architecture

A Generative Adversarial Network contains two competing neural networks:

1. **Generator** — creates synthetic observations.
2. **Discriminator** — attempts to distinguish synthetic observations from real observations.

During training, the generator progressively learns to create observations that resemble the original data.

### Generator

The generator receives a **10-dimensional random latent vector** and produces three output variables.

```text
Random Latent Vector (10)
          │
          ▼
Linear Layer: 10 → 128
Batch Normalization
LeakyReLU
          │
          ▼
Linear Layer: 128 → 128
Batch Normalization
LeakyReLU
          │
          ▼
Linear Layer: 128 → 64
Batch Normalization
LeakyReLU
          │
          ▼
Linear Layer: 64 → 3
Sigmoid
          │
          ▼
Depth | Resistivity | Su
```

### Discriminator

The discriminator receives the three geotechnical/geophysical variables and determines whether each observation is real or generated.

```text
Depth | Resistivity | Su
          │
          ▼
Linear Layer: 3 → 128
LeakyReLU
Dropout
          │
          ▼
Linear Layer: 128 → 128
LeakyReLU
Dropout
          │
          ▼
Linear Layer: 128 → 64
LeakyReLU
Dropout
          │
          ▼
Linear Layer: 64 → 1
          │
          ▼
Real / Synthetic
```

---

## Data Preprocessing

The model uses the following three variables from the training dataset:

| Variable | Description                     |
| -------- | ------------------------------- |
| `Depth`  | Depth               |
| `Res`    | Electrical resistivity          |
| `Su_kpa` | Undrained shear strength in kPa |

Before GAN training, the variables are normalized using:

```python
MinMaxScaler()
```

from Scikit-learn.

Normalization transforms the variables to a consistent numerical scale, which helps stabilize GAN training.

Random seeds are also specified for NumPy and PyTorch to improve reproducibility.

---

## Model Training

The GAN is implemented using **PyTorch**.

Key training parameters include:

| Parameter                  |  Value |
| -------------------------- | -----: |
| Latent dimension           |     10 |
| Batch size                 |     64 |
| Training epochs            |  4,000 |
| Learning rate              | 0.0002 |
| Optimizer                  |   Adam |
| Generator output variables |      3 |
| Real label                 |    0.9 |
| Fake label                 |    0.0 |

The model uses:

* Adam optimization
* Binary Cross Entropy with Logits Loss
* Label smoothing
* Batch normalization
* Dropout
* LeakyReLU activation
* Learning-rate scheduling
* Custom neural-network weight initialization

The learning rate is reduced during training using a PyTorch `StepLR` scheduler.

---

## Synthetic Data Generation

After training, the generator is used to create:

**20,000 synthetic observations**

Random vectors are sampled from the latent space and passed through the trained generator.

The generated values are then transformed back to their original measurement scale using the fitted `MinMaxScaler`.

The resulting synthetic dataset contains:

```text
Depth
Res
Su_kpa
```

---

## Synthetic Data Evaluation

The similarity between the original and synthetic datasets is evaluated using statistical and visual approaches.

### Kolmogorov-Smirnov Test
The notebook implements a two-sample **Kolmogorov-Smirnov (KS) test** to compare the distributions of the original and synthetic variables.
The comparison is performed independently for:

* Depth
* Resistivity
* Undrained shear strength

The KS statistic provides a measure of the difference between the empirical distributions of the original and synthetic observations.

---

## Visualization
The synthetic dataset is also evaluated visually using pairwise plots.
The visualizations include:

* Scatter plots
* Kernel Density Estimation (KDE)
* Histograms
* Pairwise relationships among Depth, Resistivity, and Su

These plots help assess whether important relationships among the variables are represented in the synthetic dataset.

---

## Libraries

The project uses:

* **Python**
* **PyTorch**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **SciPy**
* **Matplotlib**
* **Seaborn**

---

## Repository Structure

```text
GAN-Synthetic-Geotechnical-Data/
│
├── README.md
│
├── GAN_Synthetic_Geotechnical_Data.ipynb
│
├── requirements.txt
│
├── data/
│   └── sample_data.csv
│
├── outputs/
│   ├── synthetic_data.csv
│   └── figures/
│
└── paper/
    └── Arowoogun_et_al_2026.pdf
```

---

## Running the Notebook

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/GAN-Synthetic-Geotechnical-Data.git
```

Navigate to the project folder:

```bash
cd GAN-Synthetic-Geotechnical-Data
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Open the Jupyter Notebook:

```bash
jupyter notebook GAN_Synthetic_Geotechnical_Data.ipynb
```

The notebook can also be opened and executed using **Google Colab**.

---

## Required Python Packages

The main dependencies are:

```text
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
torch
jupyter
```

These dependencies can be stored in a `requirements.txt` file.

---

## Potential Applications

GAN-generated synthetic geotechnical data may support research involving:
* Geotechnical data augmentation
* Machine-learning model development
* Geophysical-to-geotechnical property prediction
* Data-scarce areas
* Undrained shear-strength prediction

---

## Citation

If you use this repository or methodology in academic work, please cite the associated publication:

```text
Arowoogun, K., Grote, K., & Maurer, J. (2026).Predicting undrained shear strength from Towed-TEM resistivity and sparse CPT data using machine learning models.
Journal of Applied Geophysics, 254, 106500. https://doi.org/10.1016/j.jappgeo.2026.106500
```

---

## Acknowledgment

This repository contains the GAN-based synthetic-data-generation component associated with the research described in the accompanying publication.

The GAN methodology was developed based on the foundational Generative Adversarial Network framework introduced by Goodfellow et al. (2014).

---

## Author

**Kolawole Arowoogun**

Research areas include geotechnical engineering, geophysics, machine learning, GIS, climate and environmental applications, and data-driven modeling.



