# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Dataset
data = {
    'Age': [25,30,45,35,22,40,50,23,31,48,55,60],
    'Income': [30000,40000,70000,50000,25000,65000,80000,27000,42000,72000,90000,100000],
    'Loan_Amount': [10000,15000,20000,18000,8000,25000,30000,7000,16000,22000,35000,40000],
    'Credit_Score': [0,0,1,1,0,1,1,0,0,1,1,1]   # 0 = Bad, 1 = Good
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features and Target
X = df[['Age','Income','Loan_Amount']]
y = df['Credit_Score']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create Logistic Regression model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=[0,1])
print("Confusion Matrix:")
print(cm)

# Display predictions
print("Actual Values:", list(y_test))
print("Predicted Values:", list(y_pred))
