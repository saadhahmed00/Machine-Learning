import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

digits = datasets.load_digits()

print(digits)
print(digits.DESCR)
print(digits.data)
print(digits.target)

plt.gray()
plt.matshow(digits.images[100])
plt.show()

print(digits.target[100])
###############################################
# Figure size (width, height)

fig = plt.figure(figsize=(6, 6))

# Adjust the subplots 

fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# For each of the 64 images

for i in range(64):

    # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position

    ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])

    # Display an image at the i-th position

    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')    
    
    # Label the image with the target value

    ax.text(0, 7, str(digits.target[i]))

plt.show()

##############################################
model = KMeans(n_clusters = 10, random_state = 42)
model.fit(digits.data)

fig = plt.figure(figsize=(8,3))

fig.suptitle('Cluser Center Images', fontsize=14, fontweight='bold')

for i in range(10):
  ax = fig.add_subplot(2,5,1+i)
  ax.imshow(model.cluster_centers_[i].reshape((8,8)), cmap=plt.cm.binary)
plt.show()

new_samples = np.array([
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.15,3.05,3.81,2.88,0.00,0.00,0.00,0.00,4.98,7.62,7.23,7.62,0.87,0.00,0.00,0.00,7.24,4.91,2.35,7.62,1.52,0.00,0.00,0.00,1.65,0.69,2.94,7.62,1.52,0.00,0.00,0.00,0.00,3.30,7.15,6.70,0.29,0.00,0.00,0.00,0.00,7.16,7.61,6.46,5.33,2.72,0.00,0.00,0.00,1.51,4.64,5.33,5.33,2.74,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.27,5.86,6.09,6.09,3.79,0.00,0.00,0.00,3.04,7.62,5.76,7.62,5.36,0.00,0.00,0.00,3.04,7.62,0.81,4.95,7.59,0.58,0.00,0.00,2.51,7.62,6.00,6.28,7.62,0.60,0.00,0.00,0.00,3.98,5.33,5.33,3.69,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,1.87,4.27,4.57,4.20,0.45,0.00,0.00,2.33,7.54,7.24,6.63,7.61,3.02,0.00,0.00,5.63,6.70,0.46,0.76,7.62,3.04,0.00,0.00,4.02,3.81,0.00,0.81,7.62,3.04,0.00,0.00,0.00,0.00,0.00,3.96,7.62,2.26,0.00,0.00,0.15,4.92,5.21,7.61,6.92,1.35,0.00,0.00,0.08,5.74,7.61,7.62,7.62,7.37,0.00,0.00,0.00,0.00,0.00,0.38,1.06,1.83,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,2.03,4.42,6.03,6.86,2.85,0.00,0.00,2.91,7.60,7.24,5.64,6.48,7.62,4.40,0.00,6.85,6.16,0.30,0.00,0.20,4.56,7.61,1.79,7.62,2.20,0.00,0.00,0.00,1.57,7.62,2.28,7.62,6.23,3.48,0.98,3.50,6.67,7.54,1.50,7.62,6.29,7.62,7.61,7.62,6.21,2.34,0.00,7.62,4.70,2.01,3.04,1.59,0.00,0.00,0.00]
])

new_labels = model.predict(new_samples)
print(new_labels)

for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')