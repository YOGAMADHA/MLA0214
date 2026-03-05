# FIND-S Algorithm with detailed output

# Training data
data = [
    ['sky', 'airtemp', 'humidity', 'wind', 'water', 'forecast', 'enjoysport'],
    ['sunny', 'warm', 'normal', 'strong', 'warm', 'same', 'yes'],
    ['sunny', 'warm', 'high', 'strong', 'warm', 'same', 'yes'],
    ['rainy', 'cold', 'high', 'strong', 'warm', 'change', 'no'],
    ['sunny', 'warm', 'high', 'strong', 'cool', 'change', 'yes']
]

print("\nThe total number of training instances are :", len(data))

# Initialize hypothesis with '0'
hypothesis = ['0'] * (len(data[0]) - 1)

print("\nThe initial hypothesis is :")
print(hypothesis)

i = 0

# Apply FIND-S
for example in data[1:]:
    i += 1
    if example[-1] == "yes":   # consider only positive examples
        for j in range(len(hypothesis)):
            if hypothesis[j] == '0':
                hypothesis[j] = example[j]
            elif hypothesis[j] != example[j]:
                hypothesis[j] = '?'

    print("\nThe hypothesis for the training instance", i, "is :")
    print(hypothesis)

print("\nThe Maximally specific hypothesis for the training instance is")
print(hypothesis)
