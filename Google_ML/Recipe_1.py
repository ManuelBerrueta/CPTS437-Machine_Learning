from sklearn import tree
#! For this exercise we will use a Decision Tree

#! Inputs (Data)
#* features = [[140, "smooth"], [130, "smooth"], [150, "bumpy"], [170, "bumpy"]]
#! Outputs (Our Target(s))
#* labels = ["apple", "apple", "orange", "orange"]


# We will use ints instead of strings
#! Inputs (Data)
#* smooth = 1, and bumpy = 0
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
#! Outputs (Our Target(s))
#* apple = 0, and orange = 1
labels = [0, 0, 1, 1]

#! Decision Tree Classifier
clf = tree.DecisionTreeClassifier()

#! Training algorithm is included in the classifier object
#! Think of fit as "find patters in data"
clf = clf.fit(features, labels)

#! Testing data
print(clf.predict([[150,0]]))