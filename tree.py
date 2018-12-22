
# coding: utf-8

# In[ ]:


class Node:
    """A node in a binary decision tree"""
    
    def __init__(self, left=None, right=None, feature=None, value=None, predict=None):
        """Initialize the node with attributes."""
        self.left = left
        self.right = right
        self.feature = feature # column in which features is stored
        self.value = value # value to check against
        self.predict = predict # class to predict at this node
        
    def isLeaf(self):
        """Helper function to check if the current node is a leaf"""
        return self.left is None and self.right is None
       
    def __str__(self, depth=1):
        """ You can ignore this function, 
        but basically it helps print the node in a human-readable manner """
        if self.isLeaf():
            return "Predict: \"{:s}\"".format(str(self.predict))
        else:
            s = "if features[{:d}] != \"{:s}\" then:\n {:s} \n{:s}else:\n {:s}"
            return s.format(self.feature, 
                            str(self.value), 
                            "\t" * depth+self.left.__str__(depth+1),
                            "\t" * (depth-1),
                            "\t" * depth+self.right.__str__(depth+1))


# In[ ]:


def majority(a):
    d = {}
    for i in a:
        if i in d:
            d[i] += 1
        else:
            d[i] = 1
    return sorted(d.items(), key=lambda x: int(x[1]))[-1:][0][0]


# In[ ]:


def question_set(X):
    l = []
    l2 = []
    
    for i in range(len(X[0])):
        for j in range(len(X)):
            l.append(X[j][i])
        l2.append(set(l))
        l = []
    return l2


# In[ ]:


def split(feature, value, X, y):
    real_val = []
    real_lab = []
    diff_val = []
    diff_lab = []
    
    for i in range(len(X)):
            if X[i][feature] == value:
                real_val.append(X[i])
                real_lab.append(y[i])
            else:
                diff_val.append(X[i])
                diff_lab.append(y[i])
                
    return real_val, real_lab, diff_val, diff_lab


# In[ ]:


from math import log2

def entropy(labels):
    uniq = set(labels)
    entropy = 0
    for i in uniq:
        p = labels.count(i)/ len(labels) 
        value = p * log2(p)
        entropy += value
        
    #print(abs(entropy))
    return abs(entropy)


# In[ ]:


def IG(left, right):
    hp = len(left+right)
    ig = entropy(left+right) - ((len(left) / hp) * entropy(left)) - ((len(right) / hp) * entropy(right)) 
     
    return ig


# In[ ]:


def best_features(X, y):
        # We want to first ask about value Round in column at index 2.
    best_gain = 0
    best_feature = 0
    best_value = ''
    uniq_feat = question_set(X)
    
    for col in range(len(uniq_feat)):  # for each feature
        for val in uniq_feat[col]:
        #values = set([row[col] for row in X])  # unique values in the column
        #print(values)
        #for val in values:  # for each value
            feature = col
            value = val
            # try splitting the dataset
            true_X, true_y, false_X, false_y = split(feature, value, X, y)

            # Calculate the information gain from this split
            gain = IG(true_y, false_y)

            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_value = value 
            
    return best_gain, best_feature, best_value


# In[ ]:


def fit(X, y):
    
    gain, feature, value = best_features(X,y)
    
    if gain == 0:
        return Node(predict=majority(y))
    
    rX, ry, fX, fy = split(feature, value, X, y)
    return Node(feature=feature, value=value, left=fit(rX, ry), right=fit(fX, fy))


# In[ ]:


def predict(tree, x):
    
    if tree.isLeaf():
        return tree.predict
    if tree.value in x:
        return predict(tree.left, x)
    else:
        return predict(tree.right, x)
    
    
def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs

