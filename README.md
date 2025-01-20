# Hyperspectral Image Analysis with Deep Learning Models

This repository contains implementations and results of various deep learning models applied to hyperspectral image analysis, specifically for semantic segmentation tasks. The models include UNet, DeepLab, and Multi-Layer Perceptrons (MLPs) with different optimizations, such as MRMR feature selection and coefficient adjustments.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Models Included](#models-included)
6. [Results and Analysis](#results-and-analysis)
7. [Acknowledgments](#acknowledgments)
8. [License](#license)

---

## Project Overview

This project focuses on analyzing hyperspectral images for semantic segmentation using advanced deep learning techniques. It explores various model architectures with optimizations like feature selection and customized coefficients to enhance performance on hyperspectral data.

## Repository Structure
root/ ├── models/ │ 
      ├── deeplab.py # DeepLab model implementation │ 
      ├── deeplab_defcoef.py # DeepLab with optimized coefficients │ 
      ├── deeplab_mrmr.py # DeepLab with MRMR feature selection │ 
      ├── mlp.py # MLP model implementation │ 
      ├── mlp_deff_coef.py # MLP with optimized coefficients │ 
      ├── mlp_mrmr.py # MLP with MRMR feature selection │ 
      ├── unet.py # UNet model implementation │ 
      ├── unet_defcoef.py # UNet with optimized coefficients │ 
      ├── unet_mrmr.py # UNet with MRMR feature selection 
      ├── eda.py # Exploratory Data Analysis scripts 
      ├── R_D_paper.pdf # Related research paper 
      ├── results/ # Results of experiments (to be organized) 
      └── README.md # Documentation (this file)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/your_repo_name.git
   cd your_repo_name

pip install -r requirements.txt
