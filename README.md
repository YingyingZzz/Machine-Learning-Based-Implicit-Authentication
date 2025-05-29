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
- Computes statistical features from each segment
- Saves extracted features to `Features.csv` for use in model training

**Execution flow of learn.py:**
1. Define column names for accelerometer data (timestamp, x, y, z)
2. Load 18 segmented CSV files (each containing one walking session)
3. Trim each file to 1100 rows for uniform segment length
4. Compute magnitude from x/y/z and append as a fourth axis
5. For each segment:
   - Apply overlapping sliding window (length=100, stride=50)
   - For each window:
     - Compute statistical features on each axis (x, y, z, magnitude)
6. Aggregate all feature vectors and write to 'Features.csv'

### 2. 'learn.py' – Model Training & Evaluation
- Trains classifiers (Linear SVM, KNN, Decision Tree, Random Forest, etc.)
- Evaluates models using accuracy, precision, recall, and F1-score
- Uses 10-fold cross-validation and confusion matrix visualization for analysis

**Execution flow of learn.py:**
1. Load extracted features from 'Features.csv'
2. Split the dataset into training and testing sets (60/40 split)
3. Define a dictionary of classifiers with specific hyperparameters
4. For each classifier:
   - Train on the training set
   - Predict on the test set
   - Compute 10-fold cross-validation accuracy
   - Generate and plot a normalized confusion matrix
   - Record mean accuracy and standard deviation
5. Plot a bar chart comparing classifier performances

### 3. 'plot.py' – Visualization
- Plots comparison of classifier performance using bar charts

### 4. 'test.py' – 10-Fold Cross-Validation Evaluation
- Evaluates model performance using 10-fold cross-validation over the full feature dataset
- Computes overall accuracy, precision, recall, F1-score, and optionally confusion matrix

**Execution flow of test.py:**
1. Load Features.csv containing precomputed feature vectors
2. Perform 10-fold cross-validation:
   - Split data into 10 folds
   - For each fold:
     - Use 1 fold as test, remaining 9 as train
     - Normalize features using standard scaling
     - Train model and predict test labels
     - Collect predictions and ground truth
4. Aggregate predictions from all folds
5. Compute evaluation metrics across the full dataset
6. Optionally display or save confusion matrix

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
