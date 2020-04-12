import numpy as np
import pandas as pd
from math import log
from statistics import mean
import random
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer

#scores = cross_val_score()

def accuracy(TP, FP, TN, FN):
    P = TP + FN
    N = TN + FP
    return ((TP + TN) / (P + N))

def compute_confusion_matrix(y_test, y_pred):
    # Returns entries in confusion matrix (tp, fp, tn, fn) based on comparison of
    # predictions (y_pred) to correct (y_test).
    # Assumes two classes: 0 as positive, 1 as negative
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    n = len(y_test)
    for i in range(0,n):
        if y_test[i] == 0:
            if y_pred[i] == 0:
                tp += 1
            else:
                fn += 1
        else:
            if y_pred[i] == 1:
                tn += 1
            else:
                fp += 1
    return tp, fp, tn, fn


class BaggingClassifier():
    
    def __init__(self, classifierlist):
        self.classifierlist = classifierlist


    def build_bag(self, X, y):
        """
        Build a dataset using the bagging technique

        Args:
            X: Features
            y: labels

        Returns:
            X_train: feature data
            y_train: label data
        """

        data_set_size = len(X)
        #print(f"Len of dataset={data_set_size}")
        X_train = []
        y_train = []
        for i in range(0, data_set_size):
            j = random.randrange(0, data_set_size)
            X_train.append(X[j])
            y_train.append(y[j])
        return X_train, y_train


    def majority(self, y):
        val_dict = {}
        max_value = 0
        max_value_count = 0
        for val in y:
            if val in val_dict:
                val_dict[val] += 1
            else:
                val_dict[val] = 1
            if val_dict[val] > max_value_count:
                max_value = val
                max_value_count = val_dict[val]
        return max_value
    
    
    def majority_vote(self, y_predictions):
        y_predict = []
        #predictionsSize = len(y_predictions)\
        predictionsSize = len(y_predictions[0])
        predictionsSize = 188
        for i in range(0, predictionsSize):
            col = [x[i] for x in y_predictions]
            y_predict.append(self.majority(col))
        return y_predict
    
    
    def fit(self, X, y):
        for clf in self.classifierlist:
            X_train, y_train = self.build_bag(X,y)
            clf.fit(X_train, y_train)
        return self


    def predict(self, X):
        y_predictions = []
        for clf in self.classifierlist:
            y_predictions.append(clf.predict(X))
        return self.majority_vote(y_predictions)
        #return y_predictions

    
    def score(self,y_test, y_pred):
        tp, fp, tn, fn = compute_confusion_matrix(y_test, y_pred)
        return accuracy(tp, fp, tn, fn)


class BoostingClassifier():

    def __init__(self, classifierlist,):
        self.classifierlist = classifierlist
        self.weighted_clf_list = None
        self.alpha_list = None
        self.accuracy = []
   
    
    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=3)

        alphaList = []
        temp_y = y
        weight_control = pd.DataFrame(temp_y)
        #weight_control = {}
        weight_control['weight'] = 381 #(1 / len(y))
        #weight_control['weight'] = (381,)
        weighted_clf_list = []

        for clf in self.classifierlist:
            clf.fit(X_train, y_train, sample_weight=np.array(weight_control['weight']))

            weighted_clf_list.append(clf)

            y_predict = clf.predict(X_test)
            score = clf.score(X, y)
            weight_control['predictions'] = y_predict
            weight_control['learning'] = np.where(weight_control['predictions'] == weight_control['target'],1,0)
            weight_control['misclassified'] = np.where(weight_control['predictions'] != weight_control['target'],1,0)

            calculated_accuracy = sum(weight_control['learning'])/len(weight_control['learning'])
            misclassification = sum(weight_control['misclassified'])/len(weight_control['misclassified'])


            error = np.sum(weight_control['weights']*weight_control['misclassified'])/np.sum(weight_control['weights'])
            
            error = 1 - error
            alpha = .5 * float(log(((1 - error) / error)))
            alphaList.append(alpha)

            weight_control['weights'] *= np.exp(alpha*weight_control['misclassified'])

        self.weighted_clf_list = weighted_clf_list
        self.alpha_list = alphaList

    def predict(self, X_test, y_test):
        accuracy_List = []
        predicts = []

        for each_alpha, clf in zip(self.alpha_list, self.weighted_clf_list):
            pred = each_alpha * clf.predict(X_test)
            predicts.append(pred)

            self.accuracy.append(np.sum(np.sign(np.array(predicts),axis=0) == y_test) / len(predicts[0]))
        
        self.predict = np.sign(np.sum(np.array(predicts), axis=0))


if __name__ == "__main__":
    bc = load_breast_cancer()
    #print(bc) #*Delete_ME

    #Split Data
    X, y = bc.data, bc.target
    print(X.shape)
    print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=3)

    #TODO:
    #Possibly peform bagging, by just making data sets based on the data size..

    clf0 = DecisionTreeClassifier()
    clf1 = DecisionTreeClassifier()
    clf2 = DecisionTreeClassifier()
    clf3 = DecisionTreeClassifier()
    clf4 = DecisionTreeClassifier()
    clf5 = DecisionTreeClassifier()

    clfList = [clf0, clf1, clf2, clf3, clf4, clf5]

    #clf6 = BaggingClassifier([clf1])
    #kfold = KF
    #rating = cross_val_score(estimator=clf6, X=X, y=y, cv=3)
    #print(rating)

    rating = cross_val_score(estimator=clf0, X=X, y=y, cv=3)
    print(rating)

    
    base_score = []
    bagging_score = []
    boosting_score = []
    combination_score = []

    
    #! Base classifier alone
    clf_base = DecisionTreeClassifier()
    clf_base.fit(X_train, y_train)
    y_predicted = clf_base.predict(X_test)
    print("Base Accuracy = ", metrics.accuracy_score(y_test, y_predicted))

    #! Base classifier enhanced with bagging
    clf_bagging = BaggingClassifier(clfList)
    clf_bagging.fit(X_train,y_train)
    y_predicted = clf_bagging.predict(X_test)
    print("Bagging Accuracy = ", metrics.accuracy_score(y_test, y_predicted))

    #! Base classifier enhanced with boosting
    clf_boosting = BoostingClassifier(clfList)
    clf_boosting.fit(X,y)
    clf_boosting.predict(X_test, y_test)
    print(f"Boosting Accuracy = {clf_boosting.accuracy[-1]*100}")

    #! Base classifier enhanced with both bagging and boosting
    boost_bagging_list = []

    for i in range(0,6):
        clf_bagg = BaggingClassifier(clfList)
        clf_bagg.fit(X_train, y_train)
        y_predicted = clf_bagg.predict(X_test)
        boost_bagging_list.append(clf_bagg)
    
    for i in range(0,6):
        clf_boost = BoostingClassifier(boost_bagging_list)
        clf_boost.fit(X,y)
        clf_boost.predict(X_test, y_test)
        boost_bagging_list.append(clf_boost)

    meanList = []
    for eachCombination in boost_bagging_list:
        meanList.append(eachCombination.predict(X_test, y_test))
    print(f"Boosting & Bagging Combination = {mean(meanList)}")    
