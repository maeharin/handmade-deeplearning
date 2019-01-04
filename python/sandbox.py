import numpy as np
import matplotlib.pyplot as plt

a = np.random.uniform(-100, 100, 1000)
plt.hist(a)
plt.show()
print(np.mean(a))
print(np.std(a))

#z = (a - np.mean(a)) / np.std(a)
mean = np.mean(a)
std = np.std(a)
z = (a - mean) / std
plt.hist(z)
plt.show()

print("%.3f" % np.mean(z))
print(np.std(z))
