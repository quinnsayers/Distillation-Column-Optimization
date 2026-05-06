import numpy as np #matricies
import pandas as pd #read csv
from sklearn.model_selection import train_test_split #splitting data
from sklearn.preprocessing import StandardScaler #scaling data
from scipy.optimize import minimize #optimizing the model

df = pd.read_csv('C:/Users/quinn/OneDrive/Desktop/dataset_distill.csv', sep=";")
X = df.drop(columns="Ethanol_concentration").to_numpy()
y = df["Ethanol_concentration"].to_numpy().reshape(-1, 1) #reshape into column vector

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=33, test_size=0.2,
) #test_size is 20% of the data for testing

scaleStandard_X = StandardScaler()
scaleStandard_y = StandardScaler()

X_train = scaleStandard_X.fit_transform(X_train)
X_test = scaleStandard_X.transform(X_test)

y_train = scaleStandard_y.fit_transform(y_train)
y_test = scaleStandard_y.transform(y_test)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.sqrt(2 / n_inputs) * np.random.randn(n_inputs, n_neurons) #to avoid transpose (usually (neurons x inputs))
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

ReLU_1 = Activation_ReLU()
ReLU_2 = Activation_ReLU()
ReLU_3 = Activation_ReLU()

def ReLU_derivative(x):
    return np.where(x > 0, 1, 0)

hidden_layer_1 = Layer_Dense(20, 25)
hidden_layer_2 = Layer_Dense(25, 30)
hidden_layer_3 = Layer_Dense(30, 16)
output_layer = Layer_Dense(16, 1)

#Training the model

num_iterations = 5000

alpha = 0.005

for i in range(num_iterations):
    hidden_layer_1.forward(X_train) # output shape: (949, 25)
    ReLU_1.forward(hidden_layer_1.output)
    layer_1_out = ReLU_1.output

    hidden_layer_2.forward(layer_1_out) # output shape: (949, 30)
    ReLU_2.forward(hidden_layer_2.output)
    layer_2_out = ReLU_2.output

    hidden_layer_3.forward(layer_2_out) # output shape: (949, 16)
    ReLU_3.forward(hidden_layer_3.output)
    layer_3_out = ReLU_3.output

    output_layer.forward(layer_3_out) # output shape: (949, 1)
    final_output = output_layer.output

    output_error = final_output - y_train
    
    hidden_error_3 = ReLU_derivative(layer_3_out) * np.dot(output_error, output_layer.weights.T) #(949, 1)
    hidden_error_2 = ReLU_derivative(layer_2_out) * np.dot(hidden_error_3, hidden_layer_3.weights.T)
    hidden_error_1 = ReLU_derivative(layer_1_out) * np.dot(hidden_error_2, hidden_layer_2.weights.T)

    #partial derivatives
    hidden_1_pd = X_train[:, :, np.newaxis] * hidden_error_1[: , np.newaxis, :]
    hidden_2_pd = layer_1_out[:, :, np.newaxis] * hidden_error_2[: , np.newaxis, :]
    hidden_3_pd = layer_2_out[:, :, np.newaxis] * hidden_error_3[: , np.newaxis, :]
    output_pd = layer_3_out[:, :, np.newaxis] * output_error[:, np.newaxis, :]

    #average for total gradients
    hidden_1_gradient = np.average(hidden_1_pd, axis=0)
    hidden_2_gradient = np.average(hidden_2_pd, axis=0)
    hidden_3_gradient = np.average(hidden_3_pd, axis=0)
    total_output_gradient = np.average(output_pd, axis=0)


    #update weights and gradient descent
    hidden_layer_1.weights += - alpha * hidden_1_gradient
    hidden_layer_1.biases  += - alpha * np.average(hidden_error_1, axis=0, keepdims=True)
    hidden_layer_2.weights += - alpha * hidden_2_gradient
    hidden_layer_2.biases  += - alpha * np.average(hidden_error_2, axis=0, keepdims=True)
    hidden_layer_3.weights += - alpha * hidden_3_gradient
    hidden_layer_3.biases  += - alpha * np.average(hidden_error_3, axis=0, keepdims=True)
    output_layer.weights += - alpha * total_output_gradient
    output_layer.weights += - alpha * np.average(output_error, axis=0, keepdims=True)
    if i % 500 == 0:
        mse = np.mean((final_output - y_train) ** 2)
        print(f"Epoch {i} — MSE: {mse:.6f}")

print("Model complete...")

#For saving the weights
'''
mse = np.mean((final_output - y_train)**2)
if mse < .004:
    np.savez('model_weights.npz',
        h1_weights=hidden_layer_1.weights,
        h1_biases=hidden_layer_1.biases,
        h2_weights=hidden_layer_2.weights,
        h2_biases=hidden_layer_2.biases,
        h3_weights=hidden_layer_3.weights,
        h3_biases=hidden_layer_3.biases,
        out_weights=output_layer.weights,
        out_biases=output_layer.biases,
    )
    print("Weights saved")
'''

