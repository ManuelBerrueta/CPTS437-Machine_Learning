

#* Code to binarize the test labels
y_testlabels_bin = label_binarize(y_testlabels, neg_label=0, pos_label=1, classes=[0, 1])
y_testlabels_bin = np.hstack((1 - y_testlabels_bin, y_testlabels_bin))


y_pred = clfTree.predict_proba(X_testfeatures)[:,0] # for calculating the probability of the first class
y_pred2 = clfTree.predict_proba(X_testfeatures)[:,1] # for calculating the probability of the second class
fpr, tpr, thresholds = roc_curve(y_testlabels_bin, y_pred)
auc=auc(fpr, tpr)
print("auc for the first class", auc)

fpr2, tpr2, thresholds2 = roc_curve(y_testlabels_bin, y_pred2)
auc2=auc(fpr2, tpr2)

print("auc for the second class", auc2)

# ploting the roc curve
plt.plot(fpr,tpr)
plt.plot(fpr2,tpr2)

plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.title('Roc curve')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc="lower right")
plt.show()





############################ DID NOT WORK#################################
#y_prob = [0 for _ in range(len(y_testlabels))]
y_prob = clfTree.predict_proba(X_testfeatures)
#y_prob = y_prob[:, 1]
y_testlabels_bin = label_binarize(y_testlabels, neg_label=0, pos_label=1, classes=[0, 1])
y_testlabels_bin = np.hstack((1 - y_testlabels_bin, y_testlabels_bin))


fpr, tpr, _ = roc_curve(y_testlabels_bin, y_prob)


# plot the roc curve for the model
plt.plot(fpr, tpr, linestyle='--', label='ROC CURVE')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()


#! Plot ROC Curve
""" plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.legend(loc = 'lower right')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.tile("ROC Curve)
plt.show() """



#! This plotted but nothing but I didnt see a curve

#!Converting numpy array to int
y_test_int =y_testlabels
y_test_int[y_test_int == ''] = 0.0
y_test_int = y_test_int.astype(np.float)

# calculate the fpr and tpr for all thresholds of the classification
probs = clfTree.predict_proba(X_testfeatures)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y_test_int, preds)
print("fpr, tpr", fpr, tpr)
roc_auc = auc(fpr, tpr)


# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()





#WORKING ON THIS ONE
def get_rates(actives, scores):
    """
    :type actives: list[sting]
    :type scores: list[tuple(string, float)]
    :rtype: tuple(list[float], list[float])
    """

    tpr = [0.0]  # true positive rate
    fpr = [0.0]  # false positive rate
    nractives = len(actives)
    nrdecoys = len(scores) - len(actives)

    foundactives = 0.0
    founddecoys = 0.0
    for idx, (id, score) in enumerate(scores):
        if id in actives:
            foundactives += 1.0
        else:
            founddecoys += 1.0

        tpr.append(foundactives / float(nractives))
        fpr.append(founddecoys / float(nrdecoys))

    return tpr, fpr





clfTree = DecisionTreeClassifier(criterion="entropy")  # Unbounded
clfTree.fit(X_trainfeatures, y_traininglabels)
clfTreePrediction = clfTree.predict(X_testfeatures)
clfTreeProba = clfTree.predict_proba(X_testfeatures)
tn, fp, fn, tp = confusion_matrix(y_testlabels, clfTreePrediction).ravel()
clfTree_conf_matrix = confusion_matrix(y_testlabels, clfTreePrediction)
#print("Accuracy of Decision Tree Classifier: " + str(accuracy(tp, fp, tn, fn)))
#print("F-Measure of Decision Tree Classifier: " + str(F_Measure(tp, fp, tn, fn)))

tpr, fpr = get_rates(y_testlabels, clfTreeProba)

print("tpr, fpr", tpr, fpr)
