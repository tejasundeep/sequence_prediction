# Univariate lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# Split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    x_column, y_column = list(), list()
    for i in range(len(sequence)):
        # Find the end of this pattern
        end_ix = i + n_steps
        # Check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # Gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        x_column.append(seq_x)
        y_column.append(seq_y)

    return array(x_column), array(y_column)


# Define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# Choose a number of time steps
n_steps = 3

# Split into samples
x_column, y_column = split_sequence(raw_seq, n_steps)

# Reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
x_column = x_column.reshape((x_column.shape[0], x_column.shape[1], n_features))

# Define model
model = Sequential()
model.add(LSTM(50, activation="relu", input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")

# Fit model
model.fit(x_column, y_column, epochs=200, verbose=1)

# Demonstrate prediction
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps, n_features))
predicted_y = model.predict(x_input, verbose=0)

print(predicted_y)
