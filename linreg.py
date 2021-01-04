# Libraries to implement algos  
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("linreg-data/student.csv")
math = df["Math"]
read =df["Reading"]
write = df["Writing"]
x0 = np.ones(math.size)
df.insert(3, "Bias", x0)

# Defining features, label, and weight
X = df[["Bias", "Math", "Reading"]]
W = np.zeros(X.shape[1])
Y = df["Writing"]
# Defining our cost function

def cost_function(X,Y, W):
    return (1/(2*Y.size)) * np.sum((X.dot(W) - Y)**2)

init_cost = cost_function(X,Y,W)

# Updating our weights with batch gradient descent 
def gradient_descent(X,Y,W,l_rate, iters):
    cost_history = [0] * iters 

    for i in range(iters):
        # Gradient Descent Algorithm
        prediction = X.dot(W)
        loss = prediction - Y 
        gradient = X.T.dot(loss) / Y.size
        W = W - (l_rate*gradient)

        # Record cost history 
        cost = cost_function(X,Y,W)
        cost_history[i] = cost 

    return W, cost_history

W, cost_history = gradient_descent(X, Y, W, l_rate=0.0001, iters=1000)
print(W, cost_history[-1])
bias = -0.47889172
math = 0.09137252
reading = 0.90144884
ypred = bias + X["Math"]*math + X["Reading"]*reading
r2_score = lambda y_pred, y: 1 - (np.sum((y - y_pred)**2)) / (np.sum((y - y.mean())**2))
print(r2_score(ypred, Y))

