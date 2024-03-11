import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# first 500 N((1,2,3), I)
# next 1000 N((1,2,3), I) + N((3,2,4), I) appear alternately every 200
# last 500 N((3,2,4), I)

np.random.seed(42)
epsilon = 1
mean_1 = np.array([2, 2, 6])
mean_2 = np.array([5.5, 2, 2.5])
l = []
l.append(np.random.multivariate_normal(mean_1, np.identity(3), 500))
for i in range(5):
    l.append(np.random.multivariate_normal(mean_2, np.identity(3), 200))
    l.append(np.random.multivariate_normal(mean_1, np.identity(3), 200))
l.append(np.random.multivariate_normal(mean_2, np.identity(3), 500))
trainX = np.concatenate(l, axis=0)
border = np.random.uniform(-epsilon, epsilon, trainX.shape[0])
trainy = -38 + trainX[:, 0] * trainX[:, 0] + trainX[:, 2] * trainX[:, 2] > border

data = np.core.records.fromarrays([trainX[:, 0], trainX[:, 1], trainX[:, 2], trainy], names='x1,x2,x3,y')
df = pd.DataFrame(data)
df['timestamp'] = pd.date_range(start='1/1/2018', periods=len(df), freq='2H')
df.to_csv("case3.csv", index=None)


plt.plot(data[data['y']]['x1'], data[data['y']]['x3'], '+', color='r')
plt.plot(data[~data['y']]['x1'], data[~data['y']]['x3'], '+', color='b')
plt.show()

