import csv

# Read CSV file
with open('trainingdata.csv') as file:
    data = list(csv.reader(file))

print("\nTraining Data:")
for row in data:
    print(row)

num_attributes = len(data[0]) - 1

# Initialize S and G
S = ['0'] * num_attributes
G = [['?' for i in range(num_attributes)] for j in range(num_attributes)]

print("\nInitial Specific hypothesis S:", S)
print("Initial General hypothesis G:", G)

for i in range(1, len(data)):
    instance = data[i]

    if instance[-1] == "yes":  # Positive example
        for j in range(num_attributes):
            if S[j] == '0':
                S[j] = instance[j]
            elif S[j] != instance[j]:
                S[j] = '?'
                G[j][j] = '?'

    if instance[-1] == "no":  # Negative example
        for j in range(num_attributes):
            if S[j] != instance[j]:
                G[j][j] = S[j]
            else:
                G[j][j] = '?'

    print("\nStep", i)
    print("S:", S)
    print("G:", G)

print("\nFinal Specific Hypothesis S:", S)
print("Final General Hypothesis G:", G)
