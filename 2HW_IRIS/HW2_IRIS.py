import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X_features = iris.data[:, 0:2] #Load first 2 features of data set
X_features2 = (iris.data[:, 3:]) #Load last feature of data set
X_features = np.concatenate((X_features, X_features2), axis=1) #Join this 2 above together
X_petal_len = iris.data[:, 2:3] #Load the 3rd feature
X_features_alldata = iris.data
y_label = iris.target

# Print feature names in data set
#print(iris.feature_names[1:2, 2:])
print(iris.feature_names)

# Print the target names in the data set
print(iris.target_names)

for eachFlower in range(len(iris.target)):
    print("Ex # %d: Class Label %s | Features %s" % (eachFlower, y_label[eachFlower], X_features[eachFlower]))
    #print("Ex # %d: Class Label %s | Features %s" % (eachFlower, y_label[eachFlower], X_features2[eachFlower]))
    print("Ex # %d: Class Label %s | Features %s" % (eachFlower, y_label[eachFlower], X_petal_len[eachFlower]))
    #To test that data was split accurately
    print("Ex # %d: Class Label %s | Features %s" % (eachFlower, y_label[eachFlower], X_features_alldata[eachFlower]))



#! Split the data, keep 1 third for testing
X_trainfeatures, X_testfeatures, y_traininglabels, y_testlabels = train_test_split(X_features, y_label, test_size = .3)


#! Tree Training
from sklearn import tree
clfTree = tree.DecisionTreeClassifier()
clfTree.fit(X_trainfeatures, y_traininglabels)
import matplotlib.pyplot as plt
tree.plot_tree(clfTree.fit(X_testfeatures, y_testlabels))
plt.show()

# clfTree.predict(X_testfeatures, y_testlabels)




import matplotlib.pyplot as plt
from matplotlib import lines
from mpl_toolkits.mplot3d import Axes3D
#! Plotting the data
# you need to define the min and max values from the data
step_size = 0.05
xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, step_size),
                         np.arange(y_min, y_max, step_size),
                         np.arange(z_min, z_max, step_size))

# the colors of the plot (parameter c)
# should represent the predicted class value
# we found this linewidth to work well
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xx, yy, zz, c=c_pred, marker='s', edgecolors='k', linewidth=0.2)

# you will want to enhance the plot with a legend and axes titles
plt.show()