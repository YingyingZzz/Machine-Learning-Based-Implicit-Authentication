print(__doc__)


# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
#
# Modifed by: Yingying Zhou (2023)
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),                    # K-Nearest Neighbors (k=3)
    SVC(kernel="linear", C=0.025),              # Linear Support Vector Machine 
    SVC(gamma=2, C=1),                          # RBF-kernel Support Vector Machine
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),  # Gaussian Process with RBF kernel
    DecisionTreeClassifier(max_depth=5),        # Decision Tree (max depth = 5)
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),  # Random Forest
    MLPClassifier(alpha=1),                     # Multi-layer Perceptron (Neural Net)
    AdaBoostClassifier(),                       # AdaBoost ensemble classifier
    GaussianNB(),                               # Gaussian Naive Bayes
    QuadraticDiscriminantAnalysis()]            # Quadratic Discriminant Analysis


# Understanding: Create a synthetic binary classification dataset (2 informative features, no redundancy)
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)

# Understanding: Add random uniform noise to shift the dataset for better separation
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)

# Understanding: Store linearly separable dataset
linearly_separable = (X, y)

# Understanding: Define a list of datasets for visual comparison:
# - Two-moons dataset
# - Circular dataset
# - Linearly separable synthetic dataset
datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]

# Understanding: Create a figure object with specified dimensions for the grid of subplots
figure = plt.figure(figsize=(27, 9))

# Understanding: Initialize subplot index
i = 1
# iterate over datasets
    # preprocess dataset, split into training and test part
for ds_cnt, ds in enumerate(datasets):
    # Understanding: Unpack and preprocess dataset: standardize features and split into train/test sets
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    # Understanding: Define plot boundaries with a margin
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    # Understanding: Create a meshgrid for decision boundary visualization
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)

    # Set axis bounds
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):

        # Understanding: Create a subplot for this classifier
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

        # Understanding: Fit classifier on the training data
        clf.fit(X_train, y_train)

        # Understanding: Evaluate accuracy on the test data
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        # Understanding: Generate prediction scores for each point in the mesh grid
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)

        # Understanding: Plot decision surface
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   alpha=0.6)

        # Understanding: Set axis limits and remove tick labels
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())

        # Understanding: Set subplot title to classifier name (only for first row of datasets)
        if ds_cnt == 0:
            ax.set_title(name)

        # Understanding: Display the test accuracy score in the top-right corner of the plot
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        
        # Understanding: Move to the next subplot
        i += 1

# Understanding: Adjust layout of all subplots and show the full figure
plt.tight_layout()
plt.show()