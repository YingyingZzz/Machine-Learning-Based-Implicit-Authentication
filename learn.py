
"""
@Original author: Umair Ahmed (Thu Nov 17 23:53:32 2016)

@Modified by: Yingying Zhou (2023)
"""

"""
Execution flow of learn.py:

1. Load extracted features from Features.csv
2. Split data into training and test sets
3. Define a list of classifiers to evaluate
4. For each classifier:
   - Train on training set
   - Predict and evaluate on test set
   - Perform 10-fold cross-validation
   - Plot normalized confusion matrix
   - Record mean accuracy and standard deviation
5. Plot a bar chart comparing classifier performances
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
from sklearn.svm import SVC



#Method for computing and plotting the confusion matrix, taken and modified from sklearn's official documentation website
# Understanding: Utility function to display a confusion matrix as a color-coded heatmap.
# Adapted from the official scikit-learn documentation.
# This function is used to visualize model performance in classification tasks.
# It supports optional normalization to show percentage accuracy by class.
# Useful for evaluating multi-class classification results.
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap,aspect='auto')
    plt.title(title)
    plt.colorbar()

    # Understanding: Define axis tick marks based on number of class labels
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Understanding: Optionally normalize the confusion matrix values
    if normalize:
        cm = 100*cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Understanding: Define threshold to adjust text color visibility
    thresh = cm.max() / 2.

    # Understanding: Add text labels to each cell in the matrix
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, str(round(cm[i, j],2))+'%',
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    # Understanding: Set axis labels
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

#Method for labelling the bar chart bars, taken from Matplotlib documentation
# Understanding: Utility function to add numeric labels to each bar in a bar chart.
# Adapted from the Matplotlib documentation.
def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()

        # Understanding: Place the label slightly above the bar
        ax.text(rect.get_x() + rect.get_width()/2.,     # Center the label horizontally
                1.01*height,                            # Position slightly above the top
                '%f' % float(height),                   # Format height value as float string                           
                ha='center', va='bottom')               # Center alignment
    
        
        
    
    '''
     Main code begins from here
    '''
# Understanding: Load the feature dataset using comma as delimiter and skip the header row
# Each row in the dataset represents one windowed sample with its extracted features.
# Column 0 contains the label (user/segment index), columns 1–N contain feature values.
dataset = np.loadtxt('d:\桌面\code\Features\Features.csv', delimiter=",",skiprows=1)


#Note on Numpy array splicing
#our_array[a:b,c:d]  a and b = rows index of array && c and d = column index of array 

# Understanding: Extract feature matrix X: all rows, columns 1 onward
# Resulting shape: (300, 56), representing 300 samples with 56 features each
X = dataset[:, 1:]                      #array of 300x56 dimensons i.e. 300 feature vectors each having 56 dimensions

# Understanding: Extract label vector y: all rows, column 0
y = dataset[:, 0]

# Understanding: Define class names corresponding to each user ID in the dataset
class_names = ['user-1','user-2','user-3','user-4','user-5','user-6','user-7','user-8','user-9','user-10','user-11','user-12','user-13','user-14','user-15','user-16','user-17','user-18']


# Understanding: Define a dictionary of classifiers to be evaluated
# Each key is the classifier name, and the value is an initialized sklearn classifier instance
classifiers = {'Baseline Classifier':DummyClassifier(),'Random Forest Classifier':RandomForestClassifier(max_depth=10, n_estimators=10, max_features=1), 'KNN':KNeighborsClassifier(10),'Decision Tree':DecisionTreeClassifier(max_depth=6),'Linear SVM':SVC(kernel="linear", C=0.025),'RBF SVM':SVC(gamma=2, C=1)}

# Understanding: Extract classifier names for display or result recording
classifiers_title = list(classifiers.keys())               

# Understanding: Create an empty NumPy array to store scores during evaluation
scores=np.empty(10)

# Understanding: Lists to hold mean accuracy and standard deviation for each classifier
means_scores=[]
stddev_scores=[]

#dividing the dataset into training and testing (training 60% and test=40%)
X_train,X_test,y_train,y_test = train_test_split (X,y,test_size=0.4)


#Performing classification using each classifier and computing the 10-Fold cross-validation on results
# Understanding: Loop through each classifier, train it, evaluate on test set, and compute 10-fold cross-validation scores
# Iterate through all classifiers to train, test, and evaluate each model.
# For each classifier:
# - Fit on training data
# - Predict on test data
# - Evaluate using 10-fold cross-validation
# - Visualize performance with normalized confusion matrix
# - Store mean accuracy and standard deviation for comparison
for i in range(classifiers.__len__()):
   
   # Understanding: Train classifier using training data
   classifiers[classifiers_title[i]].fit (X_train,y_train)

   # Understanding: Predict labels for the test set
   y_pred = classifiers[classifiers_title[i]].predict(X_test)

   # Understanding: Compute 10-fold cross-validation scores using the full dataset
   scores = cross_val_score(classifiers[classifiers_title[i]],X,y,cv=10)

   # Plot normalized confusion matrix
   # Understanding: Compute the confusion matrix between true and predicted labels on the test set
   cnf_matrix = confusion_matrix(y_test, y_pred)

   # Understanding: Set NumPy print options to display 2 decimal places
   np.set_printoptions(precision=2)

   # Understanding: Create a new figure for the confusion matrix plot
   plt.figure()

   # Understanding: Plot normalized confusion matrix with class names and classifier title
   plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,title=('Confusion matrix of ' + classifiers_title[i]))
   
   # Understanding: Calculate mean accuracy from cross-validation scores
   mean = scores.mean()

   # Understanding: Calculate standard deviation from cross-validation scores
   stdev = scores.std()

   # Understanding: Append mean score to result list
   means_scores.append(mean)

   # Understanding: Append standard deviation to result list
   stddev_scores.append(stdev)

   # Understanding: Output classifier name along with mean and standard deviation
   print("[Results For ",classifiers_title[i], "] Mean: ",mean," Std Dev: ",stdev)
   

#plotting the bar chart showing each classifier's mean and std deviation of cross-validation score
# Understanding: 
# ===== Summary Visualization =====
# Display a bar chart comparing mean cross-validation scores across classifiers
fig,ax= plt.subplots()

# Understanding: Draw bars representing mean accuracy for each classifier with error bars (standard deviation)
rect1 = ax.bar(np.arange(6),means_scores,0.2,color='gray',yerr=stddev_scores)

# Understanding: Set the title for the bar chart
ax.set_title('K-fold Cross-Validation Scores (K=10)',weight='bold')
ax.set_xticklabels(classifiers_title)

# Understanding: Set x-axis tick labels to classifier names
x_ticks = np.arange(0, 6, 1)  
ax.set_xticks(x_ticks)  
ax.set_xticklabels(classifiers_title, rotation=45)

# Understanding: Set y-axis label
ax.set_ylabel('Cross-Validation Score Means',weight='bold')

# Understanding: Add numeric value labels on top of each bar
autolabel(rect1)


#calculating the precision, recall, fscore and support for the best classifier i.e. Linear SVM
y_pred = classifiers['Linear SVM'].predict(X_test)
prec_rec_fscore_supt = precision_recall_fscore_support(y_test,y_pred)

#plotting precision,recall and fscore values for each class of users
fig,ax = plt.subplots()
x= np.arange(1,19)
ax.plot(x,prec_rec_fscore_supt[0],'o-')
ax.plot(x,prec_rec_fscore_supt[1],'o-')
ax.plot(x,prec_rec_fscore_supt[2],'o-')

ax.set_ylim(0.4,1.1)
ax.set_xlim(0,19)
ax.legend(['precision','recall','f-score'],loc='lower right')
ax.set_xlabel ('User Classes',weight='bold')
ax.set_title ('Precision, Recall and F-score values',weight='bold')

plt.xticks(x, class_names, rotation=45)
plt.show()