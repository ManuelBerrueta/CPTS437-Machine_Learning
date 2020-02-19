import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import lines
from mpl_toolkits.mplot3d import Axes3D


def read_data(fileName):
    data = np.genfromtxt(fileName, delimiter=',', dtype="str")
    print("DATA:\n", data)

    return data


print("Load DATA\n")
indata = read_data("alldata.csv")

#!Split Target Label from Data
X_features = indata[:, :-1] #Load first 2 features of data set
y_label = indata[:, -1] #Load the 3rd feature


#! Split the data, keep 1 third for testing
print("Split Data\n")
X_trainfeatures, X_testfeatures, y_traininglabels, y_testlabels = train_test_split(X_features, y_label, test_size = .3)#, random_state = 7919)

print("Target Labels Test:", y_testlabels)
'''First, write code to calculate accuracy and a macro f-measure using 3-fold
 cross validation for two classifiers: a majority classifier and a decision tree
  classifier. You can use the sklearn libraries for the classifiers but write
   your own code to perform cross validation and calculation of the
    performance measures.'''

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

""" if __name__ == "__main__":
    indata = read_data("alldata.csv") """