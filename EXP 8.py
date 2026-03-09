# Import required libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Sample dataset
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8],
    'Marks': [35, 40, 50, 55, 65, 70, 75, 85]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Independent variable (X) and dependent variable (y)
X = df[['Hours_Studied']]
y = df['Marks']

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict values
y_pred = model.predict(X_test)

# Display results
print("Actual Marks:", list(y_test))
print("Predicted Marks:", y_pred)

# Calculate accuracy using R2 score
accuracy = r2_score(y_test, y_pred)
print("Accuracy (R2 Score):", accuracy)
