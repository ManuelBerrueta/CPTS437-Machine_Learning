

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