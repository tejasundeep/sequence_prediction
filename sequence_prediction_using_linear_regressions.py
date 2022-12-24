import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dataset
time_sequence = [1, 2, 3, 4, 5]
price_sequence = [2, 4, 6, 8, 10]

"""
| x | y |
| 1 | 2 |
| 2 | 4 |
| 3 | 6 |
| 4 | 8 |
| 5 | 10|
"""

# Convert single dimension to multi-dimensional array
x_column = np.array(time_sequence).reshape(-1, 1)  # Features
y_column = np.array(price_sequence).reshape(-1, 1)  # Labels

# Splitting the dataset into train and test parts
x_train, x_test, y_train, y_test = train_test_split(
    x_column,
    y_column,
    test_size=0.3,  # 30% for test
    train_size=0.7,  # 70% for training
)

# Predict for
input = input("Enter Number: ")
input = np.array([int(input)]).reshape(-1, 1)

# Creating model
model = LinearRegression()
model.fit(x_train, y_train)

# Predicting price
predicted_y = model.predict(input)

print(list(np.concatenate(predicted_y).flat))
