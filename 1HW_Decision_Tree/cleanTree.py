import numpy as np
import math
from random import randrange

Label_Entropy = 1.0

def read_data(type):
    if type == "train":
        # ("traindata.csv", dtype="str")
        data = np.genfromtxt('traindata.csv', delimiter=',', dtype='str')
    else:
        data = np.genfromtxt("testdata.csv", delimiter=',', dtype="str")

    print("Your data\n", data)

    return data


# Entropy(S)
def entropy(p_pos, p_neg):
    pp = float(p_pos) / float(p_pos + p_neg)
    pn = float(p_neg) / float(p_pos + p_neg)

    if pp == 0:
        left = 0
    else:
        left = -pp * math.log2(pp)

    if pn == 0:
        right = 0
    else:
        right = -pn * math.log2(pn)

    feature_entropy = left + right

    return feature_entropy
    # return ( -p_pos * math.log2(p_pos) ) + ( -p_neg * math.log2(p_neg) )




def information_gain(S, A):
    '''Gain or Information Gain is the amount of information we gain by picking
    a particular attribute/feature'''
    # note that A is the entropy for a feature                                     #* Need to find the information gain for each attribute passed
    gain = S - entropy(S, A)
    return gain


class Node:
    def __init__(self, feature_name, feature, leftNode, rightNode, data):
        self.feature_name = feature_name
        self.feature = feature
        self.leftNode = leftNode
        self.rightNode = rightNode
        self.data = data


class Leaf:
    def __init__(self, value, count, data):
        self.value = value
        self.count = count
        self.data = data


class Feature:
    def __init__(self, pposName, pnegName, feature_name, ppos, pneg, my_entropy):
        self.pposName = pposName
        self.pnegName = pnegName
        self.feature_name = feature_name
        self.ppos = ppos
        self.pneg = pneg
        self.my_entropy = float(my_entropy)


def count_feature_vals(in_feature, in_data):
    for each_row in in_data:
        for eachVal in each_row:
            if in_feature.pposName == eachVal:
                in_feature.ppos += 1
            elif in_feature.pnegName == eachVal:
                in_feature.pneg +=1


def find_next_feature(in_data, featureArray):
    curr_entropy = 2
    featName = None
    countHigh = 0
    sameHighArray = []

    for eachFeature in featureArray:
        temp_entropy = eachFeature.my_entropy

        if temp_entropy < curr_entropy:
            curr_entropy = temp_entropy
            featName = eachFeature.feature_name  #name of feature
        elif temp_entropy == curr_entropy:
            featName = eachFeature.feature_name  #name of feature
            countHigh += 1
            sameHighArray.append(eachFeature.feature_name)
    
    #* In the case there is more than one feature with the same entropy,
    #* randomly choose one
    if countHigh >= 2:
        random_select = randrange(len(sameHighArray)-1)
        featName = sameHighArray[random_select]
    

    # Print features and gains just for debugging
    for eachFeature in featureArray:
        if eachFeature.feature_name == featName:
            #if I can take out the label_entropy out of info_gain
            # or make a different function for this
            # I could also calculate the label_entropy inside info gain or independently after this function
            gain = information_gain(Label_Entropy, curr_entropy)
            print("GAIN = ")
            print(gain)
    print("next Feature = " + featName)
    return gain, featName #NOTE: By choosing the feature with the least entropy we will get the highest gain!


def build_data_set(in_data, features):
    """Grabs the data and builds feature object out of it. 
        Returns list with all the features"""
    num_of_features = len(in_data[0])
    featureArray = []

    # This loop enumerates the features, creates a feature object, and for each attribute in the feature object
    # it counts the number it appears in the data and associates it to its name
    for eachFeature in range(num_of_features):
        # Enumerates the attributes for each feature
        feature_values = set([each_row[eachFeature] for each_row in in_data])
        feature_values = list(feature_values)
        # Initilizing feature
        if len(feature_values) >= 2:
            tempFeature = Feature(feature_values[0], feature_values[1], header[eachFeature], 0, 0, 2.0)
        else:
            return 0
        # Calculate how many of each attribute
        count_feature_vals(tempFeature, in_data);
        # Calculate entropy of this attribute
        temp_entropy = entropy(tempFeature.ppos, tempFeature.pneg)
        tempFeature.my_entropy = temp_entropy
        featureArray.append(tempFeature)

    return featureArray


def init_tree(in_data, features, label):
    global Label_Entropy
    next_feature = None
    feature_array = None
    
    #*0 Build the data set
    feature_array = build_data_set(in_data, features)

    #* 1. Calculate entropy of Target Label
    for eachFeature in feature_array:
        if eachFeature.feature_name == label:
            Label_Entropy = eachFeature.my_entropy
            break    
    
    return id3_decision_tree(in_data, label, feature_array)


def id3_decision_tree(in_data, Target_Attribute, feature_array):
    gain, next_feature = find_next_feature(in_data, feature_array)

    deleteThisFeature = None
    #! Make a node with this feature, need to remove feature from data set
    for thisFeature in feature_array:
        if thisFeature.feature_name == next_feature:
            my_tree = Node(thisFeature.feature_name, thisFeature, 0, 0, in_data)
            deleteThisFeature = thisFeature
    
    leftCount = deleteThisFeature.ppos
    leftBranchValue = deleteThisFeature.pposName
    rightCount = deleteThisFeature.pneg
    rightBranchValue = deleteThisFeature.pnegName

    feature_array.remove(deleteThisFeature)
    

    #* Splits the data for the left  & right branches
    leftData = []
    rightData = []
    for eachRow in in_data:
        if leftBranchValue in eachRow:
            leftData.append(list(eachRow))
        else:
            rightData.append(list(eachRow))

    print(leftData)
    print("\n\n")
    print(rightData)

    #TODO: Need to remove the current feature we took out from each side data
    #TODO: Need to recalculate array for each branch
    for eachRow in leftData:
        if leftBranchValue in eachRow: eachRow.remove(leftBranchValue)

    for eachRow in rightData:
        if rightBranchValue in eachRow: eachRow.remove(rightBranchValue)

    print(leftData)
    print("\n\n")
    print(rightData)

    leftFeatArray = build_data_set(leftData, 0)
    rightFeatArray = build_data_set(rightData, 0)
    

    if leftFeatArray == 0:
        thisLeaf = Leaf(leftBranchValue, leftCount, leftData)
        my_tree.leftNode = thisLeaf
        return my_tree
    else:
        my_tree.leftNode = id3_decision_tree(leftData, Target_Attribute, leftFeatArray)

    if rightFeatArray == 0:
        thisLeaf = Leaf(rightBranchValue, rightCount, rightData)
        my_tree.rightNode = thisLeaf
        return my_tree
    else:
        my_tree.rightNode = id3_decision_tree(rightData, Target_Attribute, rightFeatArray)

    return my_tree


#! Traversing the tree will involve checking if type is node or leaf
if __name__ == "__main__":
    target_label = "Stolen"
    header = ["Color", "Type", "Origin", "Stolen"]
    train_data = read_data("train")
    
    mytree = init_tree(train_data, header, target_label)

    print(mytree)
