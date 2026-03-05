import math

# Training dataset [Feature1, Feature2, Class]
data = [
[1,20,'No'],
[2,21,'No'],
[3,22,'Yes'],
[4,23,'Yes'],
[5,24,'Yes']
]

# Test dataset
test = [
[1,20,'No'],
[2,22,'No'],
[3,23,'Yes'],
[4,24,'Yes']
]

classes = ['Yes','No']

# Calculate mean
def mean(numbers):
    return sum(numbers)/len(numbers)

# Calculate variance
def variance(numbers,m):
    return sum((x-m)**2 for x in numbers)/len(numbers)

# Gaussian probability
def gaussian(x,m,v):
    return (1/math.sqrt(2*math.pi*v))*math.exp(-(x-m)**2/(2*v))

# Separate data by class
separated = {}
for row in data:
    label = row[-1]
    if label not in separated:
        separated[label] = []
    separated[label].append(row[:-1])

# Calculate mean and variance for each class
summaries = {}

for label,rows in separated.items():
    summaries[label] = []
    for i in range(len(rows[0])):
        column = [row[i] for row in rows]
        m = mean(column)
        v = variance(column,m)
        summaries[label].append((m,v))

# Prediction function
def predict(row):
    probs = {}
    for label in summaries:
        probs[label] = 1
        for i in range(len(row)):
            m,v = summaries[label][i]
            probs[label] *= gaussian(row[i],m,v)
    return max(probs,key=probs.get)

# Confusion matrix variables
TP=TN=FP=FN=0

for row in test:
    actual = row[-1]
    predicted = predict(row[:-1])

    print("Actual:",actual," Predicted:",predicted)

    if actual=='Yes' and predicted=='Yes':
        TP+=1
    elif actual=='No' and predicted=='No':
        TN+=1
    elif actual=='No' and predicted=='Yes':
        FP+=1
    elif actual=='Yes' and predicted=='No':
        FN+=1

# Confusion Matrix
print("\nConfusion Matrix")
print("TP:",TP," FP:",FP)
print("FN:",FN," TN:",TN)

# Accuracy
accuracy = (TP+TN)/(TP+TN+FP+FN)
print("\nAccuracy:",accuracy*100,"%")
