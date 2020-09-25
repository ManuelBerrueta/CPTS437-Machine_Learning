from scipy.spatial import distance
import random

#! K Nearest Neighbor with K hard coded to 1, thus it doesn't appear in
#!  in the code since it will associate with the closest point

# a is a point from our training data; b is a point from our testing data
def euclidean_distance(a, b):
    return distance.euclidean(a,b)


class ScrappKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test): #NOTE: X_test is a list of lists
        predictions = []
        for row in X_test: #each row contains the features for one example
            #label = random.choice(self.y_train) #randomly pick a label from training data
            label = self.closest(row)  #! Used to keep track of the closest one
            predictions.append(label)
        return predictions

    def closest(self, row):
        #* Using the distance from our test point to the first point in the 
        #* in the trianing data as our current best distance
        best_dist = euclidean_distance(row, self.X_train[0])
        #* Using best_index to keep track of the point that is closest to our
        #* test point. This is really the index of the point in the data.
        #! We will need it later to retrieve it's label
        best_index = 0
        #* Iterate over the data and if we find a poin that is closer than
        #* our current point, then we will update the distance and index
        for i in range(1, len(self.X_train)):
            dist = euclidean_distance(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
                #! We will use the index to retrieve and return the closet label
        return self.y_train[best_index]




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

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))