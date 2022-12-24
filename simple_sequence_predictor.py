import numpy as np

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units=1, activation="linear", input_shape=[1]))
model.compile(optimizer="sgd", loss="mean_squared_error")

xs = np.array([1, 2, 3], dtype=float)
ys = np.array([10, 15, 20], dtype=float)

model.fit(xs, ys, epochs=1000)

results = model.predict([4, 5, 6])
rounded_result = [int(np.round(num)) for num in results]

print(rounded_result)
