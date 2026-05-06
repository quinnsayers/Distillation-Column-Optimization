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


data = np.load("model_weights.npz")

hidden_layer_1.weights = data['h1_weights']  #shape (20, 25)
hidden_layer_1.biases  = data['h1_biases']   #shape (1, 25)
hidden_layer_2.weights = data['h2_weights'] #shape (25, 30)
hidden_layer_2.biases  = data['h2_biases']   #shape (1, 30)
hidden_layer_3.weights = data['h3_weights']  #shape (30, 16)
hidden_layer_3.biases  = data['h3_biases']   #shape (1, 16)
output_layer.weights   = data['out_weights'] #shape (16, 1)
output_layer.biases    = data['out_biases']  #shape (1, 1)

#z_n = preactivation layer
#h_n = hidden layer
def forward(x):
    x_row = np.array(x).flatten().reshape(1, 20)

    hidden_layer_1.output = x_row @ hidden_layer_1.weights + hidden_layer_1.biases
    ReLU_1.forward(hidden_layer_1.output)
    z1 = hidden_layer_1.output
    h1 = ReLU_1.output

    hidden_layer_2.output = h1 @ hidden_layer_2.weights + hidden_layer_2.biases
    ReLU_2.forward(hidden_layer_2.output)
    z2 = hidden_layer_2.output
    h2 = ReLU_2.output
    
    hidden_layer_3.output = h2 @ hidden_layer_3.weights + hidden_layer_3.biases
    ReLU_3.forward(hidden_layer_3.output)
    z3 = hidden_layer_3.output
    h3 = ReLU_3.output             

    output_layer.output = h3 @ output_layer.weights + output_layer.biases
    out = output_layer.output

    return float(out.squeeze()), (z1, z2, z3, h1, h2, h3)

def objective(x):
    output, (z1, z2, z3, h1, h2, h3) = forward(x)

    d_out = -np.ones((1, 1))                              
    d_h3 = d_out * output_layer.weights.T              

    #Hidden layer 3
    d_z3 = d_h3 * ReLU_derivative(z3)          #(1, 16)
    d_h2 = d_z3 @ hidden_layer_3.weights.T     #1,16) @ (16,30) -> (1,30)

    #Hidden layer 2
    d_z2 = d_h2 * ReLU_derivative(z2)          #(1, 30)
    d_h1 = d_z2 @ hidden_layer_2.weights.T     #(1,30) @ (30,25) -> (1,25)

    # Hidden layer 1
    d_z1 = d_h1 * ReLU_derivative(z1)          #(1, 25)
    d_x = d_z1 @ hidden_layer_1.weights.T     #(1,25) @ (25,20) -> (1,20)

    loss = -output
    grad = d_x.flatten() #scipy needs a flat (20,) array
    return loss, grad

n_inputs = 20
x_hi = np.array([2.00, 351.42, 352.22, 354.46, 361.22, 366.72, 367.76, 367.89, 367.90, 367.91, 367.91, 367.91, 371.05, 372.89, 373.05, 22500.00, 22650.00, 350.00, 450.00, 650.00])
x_lo = np.array([1.01, 350.76, 350.77, 350.77, 350.79, 350.80, 350.82, 350.86, 350.91, 351.00, 351.16, 351.50, 352.43, 353.15, 355.08, 450.00, 600, 290, 200.00, 350.00])
x_lo_scaled = scaleStandard_X.transform(x_lo.reshape(1, -1)).flatten()
x_hi_scaled = scaleStandard_X.transform(x_hi.reshape(1, -1)).flatten()
bounds = list(zip(x_lo_scaled, x_hi_scaled))

best_x   = None
best_value = -np.inf
num_restarts = 20

for i in range(num_restarts):
    x0 = np.random.uniform(x_lo, x_hi)

    result = minimize(
        fun = objective,
        x0 = x0,
        method = 'L-BFGS-B',
        jac = True,
        bounds = bounds,
        options = {'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8}
    )

    predicted_output = float(-result.fun)
    if predicted_output > best_value:
        best_value = predicted_output
        best_x = result.x
    
    best_value_array = np.array(best_value).reshape(1, -1)

    print(f'Restart {i+1:2d}: output = {scaleStandard_y.inverse_transform(best_value_array).item(): .6f} | converged: {result.success}')

best_x_real = scaleStandard_X.inverse_transform(best_x.reshape(1, -1)).flatten()

print(f"\nBest predicted output: {scaleStandard_y.inverse_transform(best_value_array).item()}")
feature_names = df.drop(columns='Ethanol_concentration').columns.tolist()
for name, val in zip(feature_names, best_x_real):
    print(f"{name}: {val:.4f}")

print("\nRaw y range per column:")
print(f"min={y.min():.5f}  max={y.max():.5f}")

#Testing how well the model predicts ethanol concentrations
'''
for i in range(10):
    pred, _ = forward(scaleStandard_X.transform(X[i].reshape(1, -1)).flatten())
    pred = np.array(pred).reshape(-1, 1)
    print(f"Predicted: {scaleStandard_y.inverse_transform(pred).item(): .4f}  Actual: {y[i][0]:.4f}")
'''
