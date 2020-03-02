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

#attempt to randomize data
np.random.shuffle(indata)

from scipy.sparse import coo_matrix

#! Split Features & Target Label from Data
X_features = indata[:, :-1]     # Get all the data but the last column
y_label = indata[:, -1]         # Load the 3rd feature6


#rom sklearn.utils import shuffle
#X_sparse = coo_matrix(X_features)
#X_features, X_sparse, y_label = shuffle(X_features, X_sparse, y_label,  random_state=0)


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


#clfMajority = DummyClassifier()
#clfMajority.fit(X1, y1)
#clfMajorityPrediction = clfMajority.predict(X1)
#clfMajorityPrediction = clfMajority.predict_proba(X1)
#tn, fp, fn, tp = confusion_matrix(y1, clfMajorityPrediction).ravel()
#clfMajority_conf_matrix = confusion_matrix(y1, clfMajorityPrediction)
#print("Accuracy of Majority Classifier: " + str(accuracy(tp, fp, tn, fn)))
#print("F-Measure of Majority Classifier: " + str(F_Measure(tp, fp, tn, fn)))
#
#
#clfTree = DecisionTreeClassifier(criterion="entropy")  # Unbounded
#clfTree.fit(X1, y1)
##clfTreePrediction = clfTree.predict(X1)
##clfTree_predict_ptroba = clfTree.predict_proba(X1)
#clfTreePrediction = clfTree.predict_proba(X1)
#tn, fp, fn, tp = confusion_matrix(y1, clfTreePrediction).ravel()
#clfTree_conf_matrix = confusion_matrix(y1, clfTreePrediction)
#print("Accuracy of Decision Tree Classifier: " + str(accuracy(tp, fp, tn, fn)))
#print("F-Measure of Decision Tree Classifier: " + str(F_Measure(tp, fp, tn, fn)))




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

def calculate_confusion_matrix(y_predict_proba, y_label):
    tn=0
    fp=0
    fn=0
    tp=0
    sizeOfy = len(y_label)
    
    for item in range(0, sizeOfy):
        if y_label[item] == 0:
            if y_predict_proba[item] == 0:
                tp += 1
            else:
                fn += 1;
        else:
            if y_predict_proba[item] == 1:
                tn += 1
            else:
                fp += 1

    return tn, fp, fn, tp



def ROC_Curve_Plot(y_predict_proba, y_Label_Data, tn, fp, fn, tp, title):
    roc_points = {}
    for prob_threshold in np.arange(0.0, 1.0, 0.05):
        y_pred = [0 if ypp[0] >= prob_threshold else 1 for ypp in y_predict_proba]
        #tp, fp, tn, fn = confusion_matrix(y_test, y_pred)
        tpr = float(tp) / float(tp + fn)
        fpr = float(fp) / float(fp + tn)
        if fpr in roc_points:
            roc_points[fpr].append(tpr)
        else:
            roc_points[fpr] = [tpr]
    X = []
    y = []
    for fpr in roc_points:
        X.append(fpr)
        tprs = roc_points[fpr]
        avg_tpr = sum(tprs) / len(tprs)
        y.append(avg_tpr) 
        
    y.append(0.0)
    X.append(0.0)
    plt.plot(X,y)
    #plt.axis([-.1, 1.5, -.1, 1.5])
    plt.title(title)

    #plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-.2, 1.2])
    plt.ylim([-.2, 1.2])


    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


boundedTree = DecisionTreeClassifier(criterion="entropy", max_depth=2)
boundedTree.fit(X_features, y_label)
boundedPrediction = boundedTree.predict(X_features)
bounded_predict_proba = boundedTree.predict_proba(X_features)
tn, fp, fn, tp = confusion_matrix(y_label, boundedPrediction).ravel()
bounded_conf_matrix = confusion_matrix(y_label, boundedPrediction)
print("Confusion Matrix")
print("TN: %d | FP: %d | FN: %d | TP: %d" % (tn, fp, fn, tp))
print(bounded_conf_matrix)

#tn, fp, fn, tp = calculate_confusion_matrix(bounded_predict_proba, y_label)

ROC_Curve_Plot(bounded_predict_proba, y_label, tn, fp, fn, tp, "Bounded Decision Tree ROC Curve")


#! For calculating ROC Curve Points
#boundedPrediction = boundedTree.predict_proba(X_features)
#boundedPrediction = np.array(boundedPrediction)
#y_testlabels_bin = label_binarize(y_label, neg_label=0, pos_label=1, classes=[0, 1])
#y_testlabels_bin = np.hstack((1 - y_testlabels_bin, y_testlabels_bin))
## * TESTING***
#print("Proba Prediction")
#print(boundedPrediction)
#print("y_test")
#print(y_testlabels_bin)


#bounded_x, bounded_y = get_ROC_Curve_points(boundedPrediction, y_testlabels)
#points = get_ROC_Curve_points(boundedPrediction, y_testlabels)
#points = get_ROC_Curve_points(boundedPrediction, y_testlabels_bin)

#! testing new roc_curve




unboundedTree = DecisionTreeClassifier(criterion="entropy")  # Unbounded
unboundedTree.fit(X_features, y_label)
unboundedPrediction = unboundedTree.predict(X_features)
unbounded_predict_proba = unboundedTree.predict_proba(X_features)
tn, fp, fn, tp = confusion_matrix(y_label, unboundedPrediction).ravel()
unbounded_conf_matrix = confusion_matrix(y_label, unboundedPrediction)
print("Confusion Matrix")
print(tn, fp, fn, tp)
print(unbounded_conf_matrix)

#tn, fp, fn, tp = calculate_confusion_matrix(unbounded_predict_proba, y_label)
ROC_Curve_Plot(unbounded_predict_proba, y_label, tn, fp, fn, tp, "Bounded Decision Tree ROC Curve")