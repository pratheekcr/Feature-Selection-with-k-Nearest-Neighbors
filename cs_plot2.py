import numpy as np
import scipy
import matplotlib.pyplot as plt

# corresponding y axis values
feature = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]
forward= [82,98,94,90,89,89,88,88,89,89,90,88,86,91,90,89,89,89,88,89,88,88,86,88,88,88,87,88,85,85,87,86,86,86,84,85,85,84,85,85,84,81,80,80,80,79,80,77,76,69]
plt.plot(feature, forward, marker = 'o',label = "Forward Selection")

#feature1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
backward = [76,79,80,79,83,84,84,86,87,88,90,89,89,86,87,88,90,88,89,88,89,90,90,88,88,86,86,86,86,86,85,85,85,86,85,85,86,87,87,85,86,83,83,81,81,80,78,78,76,69]
plt.plot(feature, backward,  marker ='x', label = "Backward Elimination")

# naming the x axis
plt.xlabel('Number of Features ')
# naming the y axis
plt.ylabel('Accuracy(%)')

# giving a title to my graph
plt.title('Large Dataset')


plt.legend()

# function to show the plot
plt.show()

