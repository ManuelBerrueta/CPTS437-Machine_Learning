import matplotlib.pyplot as plt
import numpy as np

x = [1.0, 2.0, 3.0, 4.0, 5.0]
y = [1.0, 2.0, 1.3, 3.75, 2.25]
ypred = [0.0, 0.0, 0.0, 0.0, 0.0]

n = len(x)

alpha = 0.001  #? Learning Rate
theta0 = 0.5
theta1 = 1.0

num_iterations = 50

# Linear Regression

epochs = 0

while epochs < num_iterations:
    for i in range(n):
        ypred[i] = theta0 + theta1 * x[i]
    if epochs == 0 or epochs == (num_iterations-1)

        #Plotting 
    