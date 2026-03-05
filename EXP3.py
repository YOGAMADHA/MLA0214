import math
import operator

# Dataset
data = [
['Sunny','Hot','High','Weak','No'],
['Sunny','Hot','High','Strong','No'],
['Overcast','Hot','High','Weak','Yes'],
['Rain','Mild','High','Weak','Yes'],
['Rain','Cool','Normal','Weak','Yes'],
['Rain','Cool','Normal','Strong','No'],
['Overcast','Cool','Normal','Strong','Yes'],
['Sunny','Mild','High','Weak','No'],
['Sunny','Cool','Normal','Weak','Yes'],
['Rain','Mild','Normal','Weak','Yes'],
['Sunny','Mild','Normal','Strong','Yes'],
['Overcast','Mild','High','Strong','Yes'],
['Overcast','Hot','Normal','Weak','Yes'],
['Rain','Mild','High','Strong','No']
]

attributes = ['Outlook','Temperature','Humidity','Wind']

# Calculate entropy
def entropy(data):
    total = len(data)
    yes = sum(1 for row in data if row[-1] == 'Yes')
    no = total - yes
    if yes == 0 or no == 0:
        return 0
    p_yes = yes/total
    p_no = no/total
    return -(p_yes*math.log2(p_yes) + p_no*math.log2(p_no))

# Information Gain
def info_gain(data, attr_index):
    total_entropy = entropy(data)
    values = set([row[attr_index] for row in data])
    weighted_entropy = 0
    
    for value in values:
        subset = [row for row in data if row[attr_index] == value]
        weighted_entropy += (len(subset)/len(data)) * entropy(subset)
    
    return total_entropy - weighted_entropy

# ID3 Algorithm
def id3(data, attributes):
    labels = [row[-1] for row in data]
    
    if labels.count(labels[0]) == len(labels):
        return labels[0]
    
    if len(attributes) == 0:
        return max(set(labels), key=labels.count)
    
    gains = [info_gain(data,i) for i in range(len(attributes))]
    best_attr_index = gains.index(max(gains))
    best_attr = attributes[best_attr_index]
    
    tree = {best_attr:{}}
    
    values = set([row[best_attr_index] for row in data])
    
    for value in values:
        subset = [row for row in data if row[best_attr_index] == value]
        new_attrs = attributes[:best_attr_index] + attributes[best_attr_index+1:]
        reduced_subset = [row[:best_attr_index] + row[best_attr_index+1:] for row in subset]
        tree[best_attr][value] = id3(reduced_subset,new_attrs)
    
    return tree

# Build tree
decision_tree = id3(data,attributes)

print("\nDecision Tree:")
print(decision_tree)

# Classify new sample
def classify(tree, attributes, sample):
    attr = list(tree.keys())[0]
    value = sample[attr]
    subtree = tree[attr][value]
    
    if isinstance(subtree, dict):
        return classify(subtree, attributes, sample)
    else:
        return subtree

sample = {'Outlook':'Sunny','Temperature':'Cool','Humidity':'High','Wind':'Strong'}

print("\nNew Sample:", sample)
print("Prediction:", classify(decision_tree,attributes,sample))
