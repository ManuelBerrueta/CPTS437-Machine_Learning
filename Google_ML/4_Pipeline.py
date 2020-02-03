from sklearn import datasets
iris = datasets.load_iris()

# X = features, y = target_label
X = iris.data
y = iris.target

# Partition the data set using train_test_split
from sklearn.model_selection import train_test_split
# Here we are saying we will split the X into half and do the same with the y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)


# Training using a tree
from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)

# Prints predictions of the trained classifier as compared to the test data
#print(predictions)

# Print accuracy of the model using the target labels in our test data
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))

#Using k neighbors
from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()
my_classifier.fit(X_train, y_train)
predictions = my_classifier.predict(X_test)

print(accuracy_score(y_test, predictions))


# NOTE: we can change the classifiers quickly by just changing two lines
#       from sklearn.<The Classifier you wish to use> import <classifier>
#       my_classifier = <Classifier>

# Check out http://playground.tensorflow.org/