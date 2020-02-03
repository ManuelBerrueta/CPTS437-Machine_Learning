class ScrappKNN():
    def fit(self, X_train, y_train):
        pass

    def predict(self, X_test):
        pass



from sklearn import datasets
iris = datasets.load_iris()

# X = features, y = target_label
X = iris.data
y = iris.target

# Partition the data set using train_test_split
from sklearn.model_selection import train_test_split
# Here we are saying we will split the X into half and do the same with the y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)


#from sklearn.neighbors import KNeighborsClassifier
my_classifier = ScrappKNN()
my_classifier.fit(X_train, y_train)
predictions = my_classifier.predict(X_test)

print(accuracy_score(y_test, predictions))