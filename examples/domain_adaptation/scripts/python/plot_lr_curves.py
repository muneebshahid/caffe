import math
import matplotlib.pyplot as plt
x = []
def inv(base_lr, gamma, it, power):
    return base_lr * math.pow((1 + gamma * it), -power)
for i in range(0, 10000):
    x.append( inv(.01, .001, i, .75))
plt.plot(range(0, 10000), x)
plt.show()