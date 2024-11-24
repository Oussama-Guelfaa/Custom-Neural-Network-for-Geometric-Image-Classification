# Custom-Neural-Network-for-Geometric-Image-Classification

## Overview

This project demonstrates the implementation of a simple feedforward neural network from scratch using NumPy. The neural network is trained and tested on both synthetic and real datasets. Additionally, shape features are extracted from images for use in classification tasks.

## Project Structure
```
project/
├── custom_neural_network.py
├── train_radial_dataset.py
├── apply_to_features_data.py
├── extract_features.py
├── Images/
│   └── Edges/
│       ├── 1_Low/
│       ├── 2_High/
│       └── (image files)
├── Output/
│   ├── features_geometric.h5
│   └── labels_high-low.h5
├── features_data.xlsx
└── README.md
```

- **custom_neural_network.py**: Contains the `CustomNeuralNetwork` class implementation.
- **train_radial_dataset.py**: Tests the neural network on a synthetic radial dataset.
- **apply_to_features_data.py**: Applies the neural network to real extracted data from `features_data.xlsx`.
- **extract_features.py**: Extracts shape features from images and saves them into HDF5 files.
- **Images/**: Contains the images used for feature extraction.
- **Output/**: Stores the extracted features and labels.
- **features_data.xlsx**: Excel file with pre-extracted features (must be provided).
- **README.md**: Project documentation.

## Setup and Installation

### Prerequisites

- Python 3.6 or higher
- NumPy
- Matplotlib
- Pandas
- Scikit-learn
- Scikit-image
- H5py
- OpenCV (if needed)

### Installation

Install the required packages using pip:

```bash
pip install numpy matplotlib pandas scikit-learn scikit-image h5py openpyxl


---

Please ensure that you have all the necessary files and data before running the scripts. Modify the file paths in the scripts if your directory structure is different.

