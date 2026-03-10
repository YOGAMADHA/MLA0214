# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Dataset
data = {
    'Car_Age': [5,3,4,2,7,1,6,8,2,3],
    'Mileage': [50000,30000,40000,20000,70000,10000,65000,80000,25000,35000],
    'Price': [500000,700000,600000,800000,400000,900000,450000,350000,750000,680000]
}

df = pd.DataFrame(data)

# Features and target
X = df[['Car_Age','Mileage']]
y = df['Price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction on test data
y_pred = model.predict(X_test)

print("Actual Prices:", list(y_test))
print("Predicted Prices:", y_pred)

print("R2 Score:", r2_score(y_test, y_pred))

# New car prediction (correct way)
new_car = pd.DataFrame([[3,30000]], columns=['Car_Age','Mileage'])

predicted_price = model.predict(new_car)

print("Predicted Price for new car:", predicted_price[0])
