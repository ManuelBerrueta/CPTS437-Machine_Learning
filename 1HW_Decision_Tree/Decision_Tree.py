import numpy as np
import math
from random import randrange


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


def find_best_feature(label_entropy, node_array):
    temphigh = 0
    curr_entropy = 1
    featName = None
    countHigh = 0
    sameHighArray = []

    for each in node_array:
        temp_entropy = entropy(each.ppos, each.pneg)
        if temp_entropy < curr_entropy:
            curr_entropy = temp_entropy
            featName = each.feature  #name of feature
        if temp_entropy <= curr_entropy:
            featName = each.feature  #name of feature
            countHigh += 1
            sameHighArray.append(each.feature)
    if countHigh >= 2:
        random_select = randrange(len(node_array)-1)
        featName = sameHighArray[random_select]


    for each in node_array:
        if each.feature == featName:
            gain = information_gain(label_entropy, curr_entropy)
            print("GAIN = ")
            print(gain)
    return featName #NOTE: By choosing the feature with the least entropy we will get the highest gain!


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
            #deletenode = node
    #my_nodes.remove(deletenode)

    mytree = []

    while len(my_nodes) != 0:
    
        next_feature = find_best_feature(target_entropy, my_nodes)

        #! In here we can check the entropy...or gains

        print("Next Feature = ")
        print(next_feature)
        print("\n")
        
        for node in my_nodes:
            if node.feature == next_feature:
                #target_entropy = entropy(node.ppos, node.pneg)
                mytree.append(node)
                deletenode = node
        my_nodes.remove(deletenode)

    i = 0
    test_data = ['Red', 'SUV', 'Domestic', 'Yes']
    while i < len(mytree) - 1:
    
        #! Possible only need this block and get rid of the above while and below code
        for i,test in enumerate(test_data):
            if test == mytree[i].pposName or test == mytree[i].pnegName:
                #*Need to use majority here
                # Need a testdata entropy function that follows the nodes
                # As we calculate each entropy see if it would happen
                pass
            else:
                print("Unknown")



        
        next_feature = find_best_feature(target_entropy, mytree)

        #! In here we can check the entropy...or gains

        print("Next Feature = ")
        print(next_feature)
        print("\n")
        
        for node in mytree:
            if node.feature == next_feature:
                #target_entropy = entropy(node.ppos, node.pneg)
                mytree.append(node)
                deletenode = node
        mytree.remove(deletenode)

    
#Next We can save the node array to a different array in the order that is supposed to go


#Then when we get test data, we can just compare at each node

# Color,Type,Origin,Stolen
