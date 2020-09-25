import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
labs = 500


# The "+ 4 *" is to give them a plus or minus 4 inches of height
grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)


# Build a histogram to visualize the current feature
# Helps to see how useful the feature may be
plt.hist([grey_height, lab_height], stacked=False, color=['r', 'b'])
plt.show()