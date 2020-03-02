import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc

# Majority Classifier
from sklearn.dummy import DummyClassifier

# Tree
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
from matplotlib import lines
from mpl_toolkits.mplot3d import Axes3D

def read_data(fileName):
    data = np.genfromtxt(fileName, delimiter=',', dtype="str")
    # print("DATA:\n", data)
    return data

def accuracy(TP, FP, TN, FN):
    P = TP + FN
    N = TN + FP
    return ((TP + TN) / (P + N))


def precision(TP, FP):
    return (TP / (TP + FP))


def recall(TP, FN):
    return (TP / (TP + FN))


def F_Measure(TP, FP, TN, FN):
    P = precision(TP, FP)
    R = recall(TP, FN)
    F = ((2 * P * R) / (P + R))
    return F

# def get_TPR()

# TODO: Code to macro f-measure using 3-fold cross validation


def cross_validate(algo, data, K):
    pass


print("Load DATA\n")
indata = read_data("alldata.csv")

#! Split Features & Target Label from Data
X_features = indata[:, :-1]     # Get all the data but the last column
y_label = indata[:, -1]         # Load the 3rd feature6

#! SPLIT DATA INTO K GROUPS, 3 IN THIS CASE
X1, X2, X3 = np.split(X_features, [len(X_features)//3, (len(X_features)//3)*2]) # split X into 3 eqally sized arrays
y1, y2, y3 = np.split(X_features, [len(y_label)//3, (len(y_label)//3)*2]) # split X into 3 eqally sized arrays

#print("Features: \n", X_features)
#print("Target Labels:\n", y_label)
print("X", X1, X2, X3)
print("y", y1, y2, y3)

#! Split each group of data, keep 1 third for testing
print("Split Groups of Data\n")

X1_trainfeatures, X1_testfeatures, y1_traininglabels, y1_testlabels = train_test_split(
    X1, y1, test_size=.3, random_state = 7919)

X2_trainfeatures, X2_testfeatures, y2_traininglabels, y2_testlabels = train_test_split(
    X2, y2, test_size=.3, random_state = 7919)

X3_trainfeatures, X3_testfeatures, y3_traininglabels, y3_testlabels = train_test_split(
    X3, y3, test_size=.3, random_state = 7919)

#print("Split data:", X1_trainfeatures, X2_trainfeatures, X3_trainfeatures)





'''Third, write code to generate and plot an ROC curve
 (containing at least 10 points). Generate two ROC curves, one based on a
  decision tree classifier with a depth bound of 2 and one with an unbounded
   decision tree. You can use the sklearn predict_proba function to provide a
    probability distribution over the class values, but create your own
     ROC curve rather than using the sklearn roc_curve function. Ideally you
      would generate the ROC curve on a holdout subset of data, but for
       simplicity in this case you can build it using the entire dataset.'''

# Distribution Curve
#$ number of intstances on Y
#$ probability on x (also called threshold)
#$ feed data in classifier
#$ if data is greather than n threshol is in one if not is in the other

#TODO: ROC Curve
# 10 points mean we use 10 different thresholds
# from lets say .05 to 0.95


clfMajority = DummyClassifier()
clfMajority.fit(X1_trainfeatures, y1_traininglabels)
clfMajorityPrediction = clfMajority.predict(X1_testfeatures)
tn, fp, fn, tp = confusion_matrix(y1_testlabels, clfMajorityPrediction).ravel()
clfMajority_conf_matrix = confusion_matrix(y1_testlabels, clfMajorityPrediction)
print("Accuracy of Majority Classifier: " + str(accuracy(tp, fp, tn, fn)))
print("F-Measure of Majority Classifier: " + str(F_Measure(tp, fp, tn, fn)))


clfTree = DecisionTreeClassifier(criterion="entropy")  # Unbounded
clfTree.fit(X1_trainfeatures, y1_traininglabels)
clfTreePrediction = clfTree.predict(X1_testfeatures)
tn, fp, fn, tp = confusion_matrix(y1_testlabels, clfTreePrediction).ravel()
clfTree_conf_matrix = confusion_matrix(y1_testlabels, clfTreePrediction)
print("Accuracy of Decision Tree Classifier: " + str(accuracy(tp, fp, tn, fn)))
print("F-Measure of Decision Tree Classifier: " + str(F_Measure(tp, fp, tn, fn)))

