# Iris sample data set
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
# The data set is ordered where the 1st label starts @ 0, 2nd @ 50, 3rd @ 100
# Marking this for test data
test_idx = [0,50,100]

# Print feature names in data set
print(iris.feature_names)

# Print the target names in the data set
print(iris.target_names)

# First flower example
#print(iris.data[0])
#print(iris.target[0])

# Print data set
for eachFlower in range(len(iris.target)):
    print("Example %d: label %s, features %s" % (eachFlower, iris.target[eachFlower], iris.data[eachFlower]))

# Training data - removing the test data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# Testing data - just adding those 3 examples to test data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

#! Train Classifier
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print(test_target)
print(clf.predict(test_data))


# Vizualizing tree

tree.plot_tree(clf.fit(iris.data, iris.target))
""" from sklearn.externals.six import StringIO
import graphviz
import pydot
dot_data = tree.export_graphviz(clf, out_file=None, 
                     feature_names=iris.feature_names,  
                     class_names=iris.target_names,  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data)  
graph """

#import graphviz 
""" dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("iris") """