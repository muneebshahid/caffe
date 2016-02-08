import math
import matplotlib.pyplot as plt
x = []
def inv(base_lr, gamma, it, power):
    return base_lr * math.pow((1 + gamma * it), -power)
for i in range(0, 200000):
    x.append( inv(.0001, .0001, i, .5))
plt.plot(range(0, 200000), x)
plt.show()