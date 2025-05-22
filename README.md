# Machine-Learning-Based-Implicit-Authentication
> Machine learning-based implicit authentication using motion data. Implemented and evaluated various ML/DL models (SVM, KNN, CNN, etc.) on UCI dataset to explore user identification from behavioral patterns.

## Overview
This project is a partial replication and exploration based on the original repository:  
[User-Identification-and-Classification-From-Walking-Activity](https://github.com/theumairahmed/User-Identification-and-Classification-From-Walking-Activity) by Umair Ahmed.

The core implementation, especially `learn.py`, was originally developed by Umair Ahmed and shared via public GitHub. Under the supervision of Dr. Di Wu, I focused on understanding the full machine learning process, reproducing key experiments, and improving result visualizations. The code was re-annotated to reflect my interpretation of the implementation logic.

## Technologies
- Language: Python
- Dependencies:
  - `numpy`, `pandas`
  - `scikit-learn`
  - `matplotlib`, `seaborn`

## Components

### 1. 'Preprocessing.py' – Feature Extraction
- Loads segmented motion sensor data (CSV format)
- Extracts statistical features for model training (e.g., mean, std)
- Outputs merged dataset `Features.csv` for ML input

### 2. 'learn.py' – Model Training & Evaluation
- Trains classifiers (Linear SVM, KNN, Decision Tree, Random Forest)
- Evaluates using accuracy, precision, recall, and F1-score

### 3. 'plot.py' – Visualization
- Plots comparison of classifier performance using bar charts

### 4. 'test.py' – (Optional) Custom Tests
- Additional evaluation logic or post-hoc model testing

## Feature Extraction Workflow
The feature extraction process used in this project follows these steps:

### 1. Input Directory
Place 18 segmented motion sensor data files (e.g., `1.csv`, `2.csv`, ...) into a folder such as `Dataset/`. Each file represents a session or user and should contain raw or windowed accelerometer values (e.g., `timestamp`, `acc_x`, `acc_y`, `acc_z`).

### 2. Run Preprocessing Script
Execute `Preprocessing.py`, which:
  - Iterates over all `.csv` files in `Dataset/`
  - Computes statistical features from each file, such as:
    - Mean, standard deviation, max, min, RMS, etc.
  - Assigns a user/session ID to each record
  - Combines all results into a single aggregated file: `Features.csv`

### 3. Feature Output Format
Each row in `Features.csv` represents a single segmented input file (e.g., a walking session). The file contains 49 columns:
  - 1st column: segment index or user/session ID
  - Remaining 48 columns: statistical features extracted from accelerometer signals

Features are extracted per axis (X, Y, Z), including:
  - Mean, Standard Deviation, Max, Min
  - Root Mean Square (RMS), Mean Absolute Deviation (MAD), Energy
  - Skewness, Kurtosis, Range, etc.

> Note: The current `Features.csv` does not include column headers. Feature names are assumed based on standard accelerometer feature engineering practices. For future improvements, explicit feature labeling is recommended to enhance clarity and reproducibility.
