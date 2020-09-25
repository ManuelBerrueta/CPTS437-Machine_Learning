import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import lines
from mpl_toolkits.mplot3d import Axes3D

iris = datasets.load_iris()
X_features = iris.data[:, 0:2] #Load first 2 features of data set
X_features2 = (iris.data[:, 3:]) #Load last feature of data set
X_features = np.concatenate((X_features, X_features2), axis=1) #Join this 2 above together
X_petal_len = iris.data[:, 2:3] #Load the 3rd feature
X_features_alldata = iris.data
y_label = iris.target
#y_label = y_label[1:3]

# Print feature names in data set
#print(iris.feature_names[1:2, 2:])
#print(iris.feature_names)

# Print the target names in the data set
#print(iris.target_names)
#print(y_label)

""" for eachFlower in range(len(iris.target)):
    print("Ex # %d: Class Label %s | Features %s" % (eachFlower, y_label[eachFlower], X_features[eachFlower]))
    #print("Ex # %d: Class Label %s | Features %s" % (eachFlower, y_label[eachFlower], X_features2[eachFlower]))
    print("Ex # %d: Class Label %s | Features %s" % (eachFlower, y_label[eachFlower], X_petal_len[eachFlower]))
    #To test that data was split accurately
    print("Ex # %d: Class Label %s | Features %s" % (eachFlower, y_label[eachFlower], X_features_alldata[eachFlower])) """

#! Split the data, keep 1 third for testing
X_trainfeatures, X_testfeatures, y_traininglabels, y_testlabels = train_test_split(X_features, y_label, test_size = .3, random_state = 7919)

colors = ['r', 'g']

#! Tree Training
from sklearn import tree
clfTree = tree.DecisionTreeClassifier()
clfTree.fit(X_trainfeatures, y_traininglabels)
#tree.plot_tree(clfTree.fit(X_trainfeatures, y_traininglabels))
#clfTree_p = clfTree.predict(X_testfeatures)
#plt.show()
print("Accuracy for Decision Tree Classifier: " + str(clfTree.score(X_testfeatures, y_testlabels)))

#? PLOTING
# x = Sepal Length |  y = Sepal Width | z = Petal Width
x_min, x_max = X_trainfeatures[:, 0].min(), X_trainfeatures[:, 0].max()
y_min, y_max = X_trainfeatures[:, 1].min(), X_trainfeatures[:, 1].max()
z_min, z_max = X_trainfeatures[:, 2].min(), X_trainfeatures[:, 2].max()


# you need to define the min and max values from the data
step_size = 0.05
xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, step_size),
                         np.arange(y_min, y_max, step_size),
                         np.arange(z_min, z_max, step_size))


# the colors of the plot (parameter c)
# should represent the predicted class value
# we found this linewidth to work well
clfTreePredict = clfTree.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
c_pred = [colors[p-1] for p in clfTreePredict]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xx, yy, zz, c=c_pred, marker='s', edgecolors='k', linewidth=0.2)
ax.set_ylabel(iris.feature_names[0])
ax.set_xlabel(iris.feature_names[1])
ax.set_zlabel(iris.feature_names[3])

plt.title("Decision Tree Graph")

# Setup Legend
legend_1 = lines.Line2D([0],[0], linestyle="none", c=colors[0], marker = 's')
legend_2 = lines.Line2D([0],[0], linestyle="none", c=colors[1], marker = 's')
ax.legend([legend_1, legend_2], iris.target_names[1:3], numpoints = 1)

plt.show()







# clfTree.predict(X_testfeatures, y_testlabels)

#! Logistic Regression
from sklearn.linear_model import LogisticRegression
#clfLogRegx = LogisticRegression(random_state=0).fit(X, y)
clfLogReg = LogisticRegression()
clfLogReg.fit(X_trainfeatures, y_traininglabels)
#clfLogReg_p = clfLogReg.predict(X_testfeatures)
print("Accuracy for Logistic Regression Classifier: " + str(clfLogReg.score(X_testfeatures, y_testlabels)))

# the colors of the plot (parameter c)
# should represent the predicted class value
# we found this linewidth to work well
clfLogRegPredict = clfLogReg.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
c_pred = [colors[p-1] for p in clfLogRegPredict]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xx, yy, zz, c=c_pred, marker='s', edgecolors='k', linewidth=0.2)
ax.set_ylabel(iris.feature_names[0])
ax.set_xlabel(iris.feature_names[1])
ax.set_zlabel(iris.feature_names[3])

plt.title("Logistic Regression Graph")

# Setup Legend
legend_1 = lines.Line2D([0],[0], linestyle="none", c=colors[0], marker = 's')
legend_2 = lines.Line2D([0],[0], linestyle="none", c=colors[1], marker = 's')
ax.legend([legend_1, legend_2], iris.target_names[1:3], numpoints = 1)
plt.show()


#! KNN
from sklearn.neighbors import KNeighborsClassifier
clfKNN = KNeighborsClassifier()
clfKNN.fit(X_trainfeatures, y_traininglabels)
#clfKNN_p = clfKNN.predict(X_testfeatures)
print("Accuracy for KNN Classifier: " + str(clfKNN.score(X_testfeatures, y_testlabels)))

# the colors of the plot (parameter c)
# should represent the predicted class value
# we found this linewidth to work well
clfKNNPredict = clfKNN.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
c_pred = [colors[p-1] for p in clfKNNPredict]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xx, yy, zz, c=c_pred, marker='s', edgecolors='k', linewidth=0.2)
ax.set_ylabel(iris.feature_names[0])
ax.set_xlabel(iris.feature_names[1])
ax.set_zlabel(iris.feature_names[3])

plt.title("KNN Regression Graph")

# Setup Legend
legend_1 = lines.Line2D([0],[0], linestyle="none", c=colors[0], marker = 's')
legend_2 = lines.Line2D([0],[0], linestyle="none", c=colors[1], marker = 's')
ax.legend([legend_1, legend_2], iris.target_names[1:3], numpoints = 1)
plt.show()


#! Perceptron
from sklearn.linear_model import Perceptron
clfPerceptron = Perceptron()
clfPerceptron.fit(X_trainfeatures, y_traininglabels)
# clfPerceptron.predict(X_testfeatures)
print("Accuracy for Perceptron Classifier: " + str(clfPerceptron.score(X_testfeatures, y_testlabels)))

# the colors of the plot (parameter c)
# should represent the predicted class value
# we found this linewidth to work well
clfPerceptronPredict = clfPerceptron.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
c_pred = [colors[p-1] for p in clfPerceptronPredict]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xx, yy, zz, c=c_pred, marker='s', edgecolors='k', linewidth=0.2)
ax.set_ylabel(iris.feature_names[0])
ax.set_xlabel(iris.feature_names[1])
ax.set_zlabel(iris.feature_names[3])

plt.title("Perceptron Graph")

# Setup Legend
legend_1 = lines.Line2D([0],[0], linestyle="none", c=colors[0], marker = 's')
legend_2 = lines.Line2D([0],[0], linestyle="none", c=colors[1], marker = 's')
ax.legend([legend_1, legend_2], iris.target_names[1:3], numpoints = 1)
plt.show()



#! SVM
from sklearn import svm
clfSVM = svm.SVC()
clfSVM.fit(X_trainfeatures, y_traininglabels)
#clfSVM.predict(X_testfeatures)
print("Accuracy for SVM Classifier: " + str(clfSVM.score(X_testfeatures, y_testlabels)))

# the colors of the plot (parameter c)
# should represent the predicted class value
# we found this linewidth to work well
clfSVMPredict = clfSVM.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
c_pred = [colors[p-1] for p in clfSVMPredict]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xx, yy, zz, c=c_pred, marker='s', edgecolors='k', linewidth=0.2)
ax.set_ylabel(iris.feature_names[0])
ax.set_xlabel(iris.feature_names[1])
ax.set_zlabel(iris.feature_names[3])

plt.title("SVM Graph")

# Setup Legend
legend_1 = lines.Line2D([0],[0], linestyle="none", c=colors[0], marker = 's')
legend_2 = lines.Line2D([0],[0], linestyle="none", c=colors[1], marker = 's')
ax.legend([legend_1, legend_2], iris.target_names[1:3], numpoints = 1)
plt.show()