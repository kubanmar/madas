import pickle

with open('data/stored_carbon_oxygen_similarities.dat', 'rb') as f:
    [xs, ys] = pickle.load(f)

import matplotlib.pyplot as plt

#print(xs,ys)

plt.scatter(xs,ys, s = 50, alpha = 0.01)
ax = plt.gca()
ax.set_yscale('log')
ax.set_ylim([1e-3,1.5])
plt.figure()
plt.hist(xs, bins = 100)
plt.figure()
plt.hist(ys, bins = 50)
whateverthing = (5 * xs + 5 * ys) /10
plt.figure()
plt.hist(whateverthing, bins = 100)
plt.show()
