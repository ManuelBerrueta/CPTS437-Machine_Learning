import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

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


print("Load DATA\n")
indata = read_data("alldata.csv")

#! Split Features & Target Label from Data
X_features = indata[:, :-1]     # Get all the data but the last column
y_label = indata[:, -1]         # Load the 3rd feature6

print("Features:\n", X_features)
print("Target Labels:\n", y_label)

#! Split the data, keep 1 third for testing
print("Split Data\n")
X_trainfeatures, X_testfeatures, y_traininglabels, y_testlabels = train_test_split(
    X_features, y_label, test_size=.3)  # , random_state = 7919)

print("Target Labels Test:", y_traininglabels)

'''First, write code to calculate accuracy and a macro f-measure using 3-fold
 cross validation for two classifiers: a majority classifier and a decision tree
  classifier. You can use the sklearn libraries for the classifiers but write
   your own code to perform cross validation and calculation of the
    performance measures.'''


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


clfMajority = DummyClassifier()
clfMajority.fit(X_trainfeatures, y_traininglabels)
clfMajorityPrediction = clfMajority.predict(X_testfeatures)
tn, fp, fn, tp = confusion_matrix(y_testlabels, clfMajorityPrediction).ravel()
clfMajority_conf_matrix = confusion_matrix(y_testlabels, clfMajorityPrediction)
print("Accuracy of Majority Classifier: " + str(accuracy(tp, fp, tn, fn)))
print("F-Measure of Majority Classifier: " + str(F_Measure(tp, fp, tn, fn)))


clfTree = DecisionTreeClassifier(criterion="entropy")  # Unbounded
clfTree.fit(X_trainfeatures, y_traininglabels)
clfTreePrediction = clfTree.predict(X_testfeatures)
tn, fp, fn, tp = confusion_matrix(y_testlabels, clfTreePrediction).ravel()
clfTree_conf_matrix = confusion_matrix(y_testlabels, clfTreePrediction)
print("Accuracy of Decision Tree Classifier: " + str(accuracy(tp, fp, tn, fn)))
print("F-Measure of Decision Tree Classifier: " + str(F_Measure(tp, fp, tn, fn)))

'''Second, provide your observations on the results. Why do the two performance
 measures provide such different results? Why do the two classifiers perform
  so differently on this task?'''

''' They provide such different results because of their biases and they way
they categorize the data. They perform so different because the Majority
Classifier is really an either or classifier where literally the majority wins,
 where as the Tree will branch out to look at the possibilities of being in one
  class or another. '''


'''Third, write code to generate and plot an ROC curve
 (containing at least 10 points). Generate two ROC curves, one based on a
  decision tree classifier with a depth bound of 2 and one with an unbounded
   decision tree. You can use the sklearn predict_proba function to provide a
    probability distribution over the class values, but create your own
     ROC curve rather than using the sklearn roc_curve function. Ideally you
      would generate the ROC curve on a holdout subset of data, but for
       simplicity in this case you can build it using the entire dataset.'''


# TODO: predict_proba - How does it help?
# TODO: ROC Curve - How To


""" def get_ROC_Curve_points(y_clf_proba_predicted, y_test):
    roc_points = {}
    for prob_threshold in np.arange(0.0, 1.0, 0.05):
        #y_pred = [0 if ypp[0] >= prob_threshold else 1 for ypp in y_clf_proba_predicted]
        #tp, fp, tn, fn = compute_confusion_matrix(y_test, y_pred)
        tp, fp, fn, tp = confusion_matrix(
            y_testlabels, y_clf_proba_predicted).ravel()
        tpr = float(tp) / float(tp + fn)
        fpr = float(fp) / float(fp + tn)
        if fpr in roc_points:
            roc_points[fpr].append(tpr)
        else:
            roc_points[fpr] = [tpr]

    x1 = []
    y1 = []
    for fpr in roc_points:
        x1.append(fpr)
        tprs = roc_points[fpr]
        avg_tpr = sum(tprs) / len(tprs)
        y1.append(avg_tpr)

    #return x1, y1
    return roc_points """
def get_ROC_Curve_points(y_pred_proba, y_test):
    roc_points = {}
    for prob_threshold in np.arange(0.0, 1.0, 0.05):
        y_pred = [0 if ypp[0] >=
        prob_threshold else 1 for ypp in y_pred_proba]
        tp, fp, tn, fn = confusion_matrix(y_test, y_pred)
        tpr = float(tp) / float(tp + fn)
        fpr = float(fp) / float(fp + tn)
        if fpr in roc_points:
            roc_points[fpr].append(tpr)
        else:
            roc_points[fpr] = [tpr]
    x1 = []
    y1 = []
    for fpr in roc_points:
        x1.append(fpr)
        tprs = roc_points[fpr]
        avg_tpr = sum(tprs) / len(tprs)
        y1.append(avg_tpr)
        
    return x1, y1


boundedTree = DecisionTreeClassifier(criterion="entropy", max_depth=2)
boundedTree.fit(X_trainfeatures, y_traininglabels)
boundedPrediction = boundedTree.predict(X_testfeatures)
tn, fp, fn, tp = confusion_matrix(y_testlabels, boundedPrediction).ravel()
bounded_conf_matrix = confusion_matrix(y_testlabels, boundedPrediction)
print("Confusion Matrix")
print("TN: %d | FP: %d | FN: %d | TP: %d" % (tn, fp, fn, tp))
print(bounded_conf_matrix)
""" boundScore = boundedTree.predict_proba(X_testfeatures)
boundScore = np.array(boundScore)
print("Bounbded Score:")
print(boundScore)
 """

#! For calculating ROC Curve Points
boundedPrediction = boundedTree.predict_proba(X_testfeatures)
boundedPrediction = np.array(boundedPrediction)
y_testlabels_bin = label_binarize(y_testlabels, neg_label=0, pos_label=1, classes=[0, 1])
y_testlabels_bin = np.hstack((1 - y_testlabels_bin, y_testlabels_bin))
# * TESTING***
print("Proba Prediction")
print(boundedPrediction)
print("y_test")
print(y_testlabels_bin)

boundedPrediction = boundedTree.predict(X_testfeatures)
#bounded_x, bounded_y = get_ROC_Curve_points(boundedPrediction, y_testlabels)
#points = get_ROC_Curve_points(boundedPrediction, y_testlabels)
#points = get_ROC_Curve_points(boundedPrediction, y_testlabels_bin)
#! testing new roc_curve
x, y = get_ROC_Curve_points(boundedPrediction, y_testlabels_bin)

print("POINTS")
print(x, y)


unboundedTree = DecisionTreeClassifier(criterion="entropy")  # Unbounded
unboundedTree.fit(X_trainfeatures, y_traininglabels)
unboundedPrediction = unboundedTree.predict(X_testfeatures)
tn, fp, fn, tp = confusion_matrix(y_testlabels, unboundedPrediction).ravel()
unbounded_conf_matrix = confusion_matrix(y_testlabels, unboundedPrediction)
print("Confusion Matrix")
print(tn, fp, fn, tp)
print(unbounded_conf_matrix)
""" unboundedScore = unboundedTree.predict_proba(X_testfeatures)
unboundedScore = np.array(unboundedScore)
print("Unbounbded Score:")
print(unboundedScore)
 """

#! For calculating ROC Curve Points
unboundedPrediction = unboundedTree.predict(X_testfeatures)
#unbounded_x, unbounded_y = get_ROC_Curve_points(unboundedPrediction, y_testlabels)
points = get_ROC_Curve_points(boundedPrediction, y_testlabels)

""" print("Unbounded_x")
print(unbounded_x)
print("Unbounded_y")
print(unbounded_y)

for xpoint in unbounded_x:
    print(xpoint);
    
for ypoint in unbounded_y:
    print(xpoint); """

print("ROC POINTS TEST")
print(points)

# SKlearn roc curve for testing


""" plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.legend(loc = 'lower right')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.tile("ROC Curve)
plt.show() """
