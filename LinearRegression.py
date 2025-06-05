import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

file_path = "/Users/muskansharma/Documents/IBM/placement.csv"
df = pd.read_csv(file_path)
print("Data Sample:\n", df.head())

X = df[['cgpa']]                          
y = df['package']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('CGPA')
plt.ylabel('Placement Score')
plt.title('Linear Regression - Placement Prediction')
plt.legend()
plt.grid(True)
plt.show()