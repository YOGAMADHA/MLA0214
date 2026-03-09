# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Sample dataset
X = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
y = np.array([5,7,9,15,20,28,40,55,75,100])

# -------- Linear Regression --------
linear_model = LinearRegression()
linear_model.fit(X, y)
y_linear_pred = linear_model.predict(X)

# -------- Polynomial Regression --------
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_poly_pred = poly_model.predict(X_poly)

# -------- Accuracy Comparison --------
linear_r2 = r2_score(y, y_linear_pred)
poly_r2 = r2_score(y, y_poly_pred)

print("Linear Regression R2 Score:", linear_r2)
print("Polynomial Regression R2 Score:", poly_r2)

# -------- Visualization --------
plt.scatter(X, y, color="black", label="Actual Data")
plt.plot(X, y_linear_pred, color="blue", label="Linear Regression")
plt.plot(X, y_poly_pred, color="red", label="Polynomial Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Linear vs Polynomial Regression")
plt.show()
