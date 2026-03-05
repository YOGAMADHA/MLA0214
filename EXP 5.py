import math

# Dataset: [Height, Weight, Class]
data = [
[150,50,'Underweight'],
[160,60,'Normal'],
[170,65,'Normal'],
[180,80,'Overweight'],
[190,90,'Overweight']
]

# K value
k = 3

# New sample
test = [165,62]

# Calculate distance
distances = []

for row in data:
    d = math.sqrt((row[0]-test[0])**2 + (row[1]-test[1])**2)
    distances.append((d,row[2]))

# Sort distances
distances.sort()

# Get nearest neighbors
neighbors = distances[:k]

# Count classes
count = {}

for d,label in neighbors:
    if label in count:
        count[label] += 1
    else:
        count[label] = 1

# Prediction
prediction = max(count, key=count.get)

print("Test Data:",test)
print("Nearest Neighbors:",neighbors)
print("Predicted Class:",prediction)
