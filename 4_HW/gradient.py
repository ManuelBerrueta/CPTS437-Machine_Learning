import matplotlib.pyplot as plt
import numpy as np

x = [1.0, 2.0, 3.0, 4.0, 5.0]
y = [1.0, 2.0, 1.3, 3.75, 2.25]
ypred = [0.0, 0.0, 0.0, 0.0, 0.0] # h(x) in slides

n = len(x)

#alpha = 0.001  #* Learning Rate
#theta0 = 0.5
#theta1 = 1.0

alpha = 0.003  #* Learning Rate
theta0 = 0.5
theta1 = 1.3

num_iterations = 50

# Linear Regression
epochs = 0
while epochs < num_iterations:
    print('Epoch ', epochs)
    for i in range(n):
        ypred[i] = theta0 + theta1 * x[i]
    if epochs == 0 or epochs == (num_iterations - 1):
        #Plotting
        plt.scatter(x, y, color='red')
        plt.plot(x, ypred, color='green', linewidth=3)
        plt.xticks(())
        plt.yticks(())
        plt.show()

    error = 0
    total_diff0 = 0
    total_diff1 = 0
    for i in range(n):
        diff = y[i] - ypred[i]      # y - h(x)
        total_diff0 += diff         # Update Theta_0 by diff
        total_diff1 += diff + x[i]  # Update Theta_1 by diff * x
        error += diff**2            # sum of squared difference
    error /= 2*len(y)
    print('error', error, 'Theta0', theta0, 'Theta1', theta1)

    theta0 += alpha * total_diff0
    theta1 += alpha * total_diff1
    epochs += 1