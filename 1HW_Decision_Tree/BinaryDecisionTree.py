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


#! This function is the original feature function.
#! This could possibly be ran inside a bigger entropy function with the original
#! entropy function inside. This bigger entropy function will take all the data
#! as passed on that iteration, and figure out which feature to choose
#! This needs to be done in such a way that the data is split by the feature
#! chosen for that iteration
def count_feature_vals(in_feature, in_data):
    for each_row in in_data:
        for eachVal in each_row:
            if in_feature.pposName == eachVal:
                in_feature.ppos += 1
            elif in_feature.pnegName == eachVal:
                in_feature.pneg +=1


def data_entropy(data):
    #* Refer to count_feature_vals above
    #! The whole data set will come in here
    #! We need to figure out, which data gives us our next feature by 
    #! calculating the entropy of each subset of data
    #! Here we can print out the results of each entropy of each feature
    #! We can keep track of each entropy produced
    #! After we have iterated to all the features in this subdata, we can review
    #! all the entropys we are tracking and choose our next feature based on that

    
    return


def information_gain(S, A):
    '''Gain or Information Gain is the amount of information we gain by picking
    a particular attribute/feature'''
    # note that A is the entropy for a feature                                     #* Need to find the information gain for each attribute passed
    gain = S - entropy(S, A)
    return gain


class Feature:
    def __init__(self, pposName, pnegName, feature_name, ppos, pneg, my_entropy,my_gain):
        self.pposName = pposName
        self.pnegName = pnegName
        self.feature_name = feature_name
        self.ppos = ppos
        self.pneg = pneg
        self.my_entropy = float(my_entropy)
        self.my_gain = float(my_gain)


class Node:
    def __init__(self, feature_name, feature, left, right, leaf, depth):
        self.feature_name = feature_name
        self.feature = feature
        self.left_child = left
        self.right_child = right
        self.leaf = leaf
        self.depth = depth


#Brainstorming:
#* Split the data based on the target label, then send it to the count_feature_vals
