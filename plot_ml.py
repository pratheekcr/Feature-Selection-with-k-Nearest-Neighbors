import numpy as np
import scipy
import matplotlib.pyplot as plt

# corresponding y axis values
feature = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
forward= [70.2247, 92.6966, 96.62921, 96.62921, 97.19011, 97.7528, 98.3146, 98.8764, 98.31460, 98.87640, 97.19101, 96.06741, 95.50561]
plt.plot(feature, forward, marker = 'o',label = "Forward Selection")

feature1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
backward = [66.2921, 92.6966, 92.6966, 94.9438, 95.5056, 97.1910, 96.6292, 97.752, 98.3146, 97.7528, 97.1910, 97.1910, 95.5056]
plt.plot(feature1, backward,  marker ='x', label = "Backward Elimination")

# naming the x axis
plt.xlabel('Number of Features ')
# naming the y axis
plt.ylabel('Accuracy(%)')

# giving a title to my graph
plt.title('Wine Dataset')


plt.legend()

# function to show the plot
plt.show()

