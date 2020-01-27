import numpy as np
import math


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
    pp = p_pos / (float)(p_pos + p_neg)
    pn = p_neg / (float)(p_pos + p_neg)

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

# Gain or Information Gain is the amount of information we gain by picking
# a particular attribute/feature
# Information Gain(S, A):


def information_gain(S, A):
    # note that A is the entropy for a feature                                     #* Need to find the information gain for each attribute passed
    gain = S - entropy(S, A)
    return gain


""" def enum_features(indata):
    features_with_count = []{}
    for eachRow in indata:
        for eachFeat in eachRow:

        if tempFeature not in features_with_count:
            features_with_count[tempFeature] = 0
        else:
            features_with_count[tempFeature] += 1
    return features_with_count """


def split_data(indata, features):
    node_array = []
    for col in range(len(indata[0])):
        #enumerate the values of each features
        feat_values = set([features[col] for features in train_data])
        feat_values = list(feat_values)
        print(feat_values)
        

        tempNode = node(feat_values[0], feat_values[1], features[col], 0, 0)
        node_array.append(tempNode)

        print(tempNode.feature)
    return node_array

def count_feature_vals(in_node, in_data):
    for each_row in in_data:
        for eachVal in each_row:
            if in_node.pposName == eachVal:
                in_node.ppos += 1
            elif in_node.pnegName == eachVal:
                in_node.pneg +=1


def find_best_feature(data):
    temphigh = 0
    curr_entropy = 0
    featName = None
    for each in data:
        temp_entropy = entropy(each.ppos, each.pneg)
        if temp_entropy > curr_entropy:
            curr_entropy = temp_entropy
            featName = each.feature  #name of feature
    return featName


    return 0

def init_tree(label, indata):
    return 0


def id3_decision_tree(Examples, Target_Attribute, Features):
    return 0

def get_values(data):
    return 0

class node:
    def __init__(self, pposName, pnegName, feature, ppos, pneg):
        self.pposName = pposName
        self.pnegName = pnegName
        self.feature = feature
        self.ppos = ppos
        self.pneg = pneg

#def delete_node()


if __name__ == "__main__":

    print(entropy(5, 9))
    print(entropy(6, 4))

    

    header = ["Color", "Type", "Origin", "Stolen"]
    # names=['Color','Type','Origin','Stolen'],
    train_data = read_data("train")


    my_nodes = split_data(train_data, header)
    for node in my_nodes:
        count_feature_vals(node, train_data)


    #1 Calculate Entropy of Target
    target_label = "Stolen"
    target_entropy = 0
    deletenode = None
    for node in my_nodes:
        if node.feature == "Stolen":
            target_entropy = entropy(node.ppos, node.pneg)
            deletenode = node
    my_nodes.remove(deletenode)

    
    next_feature = find_best_feature(my_nodes)

    print(next_feature)


            #partition values


    # print(enum_features(train_data))

# Color,Type,Origin,Stolen
