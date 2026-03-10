# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Sample dataset
data = {
    'Area': [1000,1500,1800,2000,2200,2500,2700,3000,3200,3500],
    'Bedrooms': [2,3,3,4,4,4,5,5,5,6],
    'Age': [10,8,6,5,7,4,3,2,1,1],
    'Price': [200000,250000,270000,300000,320000,350000,370000,400000,420000,450000]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[['Area','Bedrooms','Age']]
y = df['Price']

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train,y_train)

# Predict house prices
y_pred = model.predict(X_test)

# Display results
print("Actual Prices:", list(y_test))
print("Predicted Prices:", y_pred)

# Accuracy
print("R2 Score:", r2_score(y_test,y_pred))

# Predict price for new house
new_house = pd.DataFrame([[2100,4,5]],columns=['Area','Bedrooms','Age'])

predicted_price = model.predict(new_house)

print("Predicted House Price:", predicted_price[0])
