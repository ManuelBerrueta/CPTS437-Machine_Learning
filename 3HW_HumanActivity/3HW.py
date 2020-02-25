import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve

# Majority Classifier
from sklearn.dummy import DummyClassifier

# Tree
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
from matplotlib import lines
from mpl_toolkits.mplot3d import Axes3D


def read_data(fileName):
    data = np.genfromtxt(fileName, delimiter=',', dtype="str")
    #print("DATA:\n", data)
    return data


print("Load DATA\n")
indata = read_data("alldata.csv")

#! Split Features & Target Label from Data
X_features = indata[:, :-1]     # Get all the data but the last column
y_label = indata[:, -1]         # Load the 3rd feature6

print("Features:\n", X_features)
print("Target Labels:\n", y_label)

#! Split the data, keep 1 third for testing
print("Split Data\n")
X_trainfeatures, X_testfeatures, y_traininglabels, y_testlabels = train_test_split(X_features, y_label, test_size = .3)#, random_state = 7919)

print("Target Labels Test:", y_traininglabels)
'''First, write code to calculate accuracy and a macro f-measure using 3-fold
 cross validation for two classifiers: a majority classifier and a decision tree
  classifier. You can use the sklearn libraries for the classifiers but write
   your own code to perform cross validation and calculation of the
    performance measures.'''

clfMajority = DummyClassifier()
clfMajority.fit(X_trainfeatures, y_traininglabels)

#TODO: Code to Calculate Accuracy

#TODO: Code to macro f-measure using 3-fold cross validation


boundedTree = DecisionTreeClassifier(criterion="entropy", max_depth=2)
boundedTree.fit(X_trainfeatures, y_traininglabels)
boundScore = boundedTree.predict_proba(X_testfeatures)
boundScore = np.array(boundScore)
print("Bounbded Score:")
print(boundScore)

unboundedTree = DecisionTreeClassifier(criterion="entropy") #Unbounded
unboundedTree.fit(X_trainfeatures, y_traininglabels)
unboundedScore = unboundedTree.predict_proba(X_testfeatures)
unboundedScore = np.array(unboundedScore)
print("Unbounbded Score:")
print(unboundedScore)

'''Second, provide your observations on the results. Why do the two performance
 measures provide such different results? Why do the two classifiers perform
  so differently on this task?'''


'''Third, write code to generate and plot an ROC curve
 (containing at least 10 points). Generate two ROC curves, one based on a
  decision tree classifier with a depth bound of 2 and one with an unbounded
   decision tree. You can use the sklearn predict_proba function to provide a
    probability distribution over the class values, but create your own
     ROC curve rather than using the sklearn roc_curve function. Ideally you
      would generate the ROC curve on a holdout subset of data, but for
       simplicity in this case you can build it using the entire dataset.'''


# SKlearn roc curve for testing






""" plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show() """