import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, make_circles, make_classification

# Understanding: Load the 'Features.csv' file into the variable `dataset`,
# using comma as delimiter and skipping the header row
dataset = np.loadtxt('d:\桌面\code\Features\Features.csv', delimiter=",",skiprows=1)

# Understanding: Extract the feature matrix X - all rows, columns 1 onward
# Shape: (num_samples, num_features)
X = dataset[:, 1:]                      #array of 300x56 dimensons i.e. 300 feature vectors each having 56 dimensions

# Understanding: Extract the label vector Y - all rows, first column (segment or user index)
# Shape: (num_samples,)
Y = dataset[:, 0]

# Understanding: Set print options to disable output truncation for large arrays
np.set_printoptions(threshold='nan')

# Understanding: Print feature matrix and label vector
print X
print"*****************************************"
print Y