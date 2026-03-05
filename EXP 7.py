import math

# Training dataset (Hours studied vs Pass/Fail)
X = [1, 2, 3, 4, 5, 6]
Y = [0, 0, 0, 1, 1, 1]

# Initialize weights
w = 0
b = 0
learning_rate = 0.1
epochs = 1000

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# Training using Gradient Descent
for i in range(epochs):
    dw = 0
    db = 0
    n = len(X)
    
    for j in range(n):
        z = w * X[j] + b
        y_pred = sigmoid(z)
        
        dw += (y_pred - Y[j]) * X[j]
        db += (y_pred - Y[j])
    
    dw = dw / n
    db = db / n
    
    w = w - learning_rate * dw
    b = b - learning_rate * db

# Prediction
def predict(x):
    z = w * x + b
    y_pred = sigmoid(z)
    
    if y_pred >= 0.5:
        return 1
    else:
        return 0

# Testing
test = 3.5
result = predict(test)

print("Weight:", w)
print("Bias:", b)
print("Prediction for", test, "=", result)

if result == 1:
    print("Student will PASS")
else:
    print("Student will FAIL")
