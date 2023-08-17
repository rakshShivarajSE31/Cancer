import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

x = np.array([ 0, 0.53, 1.05, 1.58, 2.11, 2.63, 3.16, 3.68, 4.21, 4.74, 5.26, 5.79, 6.32, 6.84])

y = np.array([ 0, 0.51, 0.87, 1. , 0.86, 0.49, -0.02, -0.51, -0.88,  -1. , -0.85, -0.47, 0.04, 0.53])

## Creating a figure to plot the graph.
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel('X data')
ax.set_ylabel('Y data')
ax.set_title('Relationship between variables X and Y')
plt.show()# display the graph
### if %matplotlib inline has been invoked already, then plt.show() is automatically invoked and the plot is displayed in the same window.