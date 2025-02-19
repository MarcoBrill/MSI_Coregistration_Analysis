# MSI and Histological Image Coregistration

This repository provides a Python script for coregistering Mass Spectrometry Imaging (MSI) data with histological images (e.g., H&E), creating a 3D model of a tumor, and analyzing molecules using modern computer vision techniques.

## Inputs
1. **MSI Data**: A numpy array or .imzML file containing MSI data.
2. **Histological Image**: A high-resolution H&E stained image in .png or .tiff format.
3. **Tumor Mask**: A binary mask of the tumor region in .png format.

## Outputs
1. **Coregistered Image**: A registered image combining MSI and histological data.
2. **3D Tumor Model**: An STL file representing the 3D structure of the tumor.
3. **Molecule Analysis**: Heatmaps or clustered images of molecular distributions.

## Requirements
- Python 3.8+
- Libraries: numpy, opencv-python, scikit-image, scipy, scikit-learn, matplotlib, stl

## Installation
```bash
pip install -r requirements.txt
