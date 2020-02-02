import collections
import numpy as np


# Read a data file in csv format, separate into features and class arrays
def read_data(type):
   if type == 'train':
      data = np.loadtxt(fname='traindata.csv', delimiter=',')
   else:
      data = np.loadtxt(fname='testdata.csv', delimiter=',')
   print("Your data", data)
   X = data[:,:-1]   # features are all values but the last on the line
   y = data[:,-1]    # class is the last value on the line
   print("X", X, "y", y)
   return X, y


# The simple majority classifier determines the most common class label
# and labels all instances with that class value
def simple_majority_train(X, y):
   majority_class = collections.Counter(y).most_common(1)[0][0]
   print(majority_class)
   return majority_class


# Classify test instances based on majority label
def simple_majority_test(X, y, majority_class):
   total = len(y)
   true_positive = 0
   false_positive = 0
   true_negative = 0
   false_negative = 0

   for i in range(total):    # evaluate each test instance
      label = majority_class # not really needed, just illustrates point
      if label == 0.0:       # majority label is negative
         if y[i] == 0.0:     # this is a negative instance
            true_negative += 1
         else:               # this is a positive instance
            false_negative += 1
      else:                  # majority label is positive (label == 1.0)
         if y[i] == 0.0:     # this is a negative instance
            false_positive += 1
         else:               # this is a positive instance
            true_positive += 1
   report_statistics(total, true_positive, false_positive,
                     true_negative, false_negative)


def report_statistics(total, tp, fp, tn, fn):
   print("total", total, "tp", tp, "fp", fp, "tn", tn, "fn", fn)


# Train and test simple majority classifier
if __name__ == "__main__":
   # read training data and use to train classifier
   X, y = read_data('train')
   #print(collections.Counter(y))
   majority_class = simple_majority_train(X, y)

   # read test data and test classifier
   #X, y = read_data('test')
   #simple_majority_test(X, y, majority_class)
