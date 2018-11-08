import pickle

with open('data/stored_diamond_plat_similarities.dat', 'rb') as f:
    [xs, ys] = pickle.load(f)

import matplotlib.pyplot as plt

#print(xs,ys)

plt.scatter(xs,ys)
plt.show()
