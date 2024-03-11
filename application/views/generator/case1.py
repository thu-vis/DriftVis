import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

'''
data: x1, x2, x3
[0,1000), [1000,2000), [2000,3000), [3000,4000), [4000,5000) are five gaussian distribution
[5000, 5200). [5200, 5400). [5400, 5600) are three test distribution
'''


np.random.seed(42)
n = 1000
n_test = 200
epsilon = 0.5

train1 = np.random.multivariate_normal(np.array([0, 5]), np.array([[0.5, 0], [0, 1]]), n)
train2 = np.random.multivariate_normal(np.array([1, 3]), np.array([[1, -0.2], [-0.2, 1]]), n)
train3 = np.random.multivariate_normal(np.array([3, 1]), np.array([[0.5, 0], [0, 0.5]]), n)
train4 = np.random.multivariate_normal(np.array([5, 3]), np.array([[1, 0.2], [0.2, 1]]), n)
train5 = np.random.multivariate_normal(np.array([6, 5]), np.array([[1, 0], [0, 2]]), n)

trainX = np.concatenate((train1, train2, train3, train4, train5), axis=0)
border = np.random.uniform(-epsilon, epsilon, trainX.shape[0])
trainy = (trainX[:, 0] - 3) * (trainX[:, 0] - 3) / 2 + 1 - trainX[:, 1] > border

test1 = np.random.multivariate_normal(np.array([0, 5]), np.array([[0.5, 0], [0, 1]]), n_test)
test2 = np.random.multivariate_normal(np.array([5, 3]), np.array([[1, 0.2], [0.2, 1]]), n_test)
test3 = np.random.multivariate_normal(np.array([3, 1]), np.array([[0.5, 0], [0, 0.5]]), n_test)
testX = np.concatenate((test1, test2, test3), axis=0)
testy = (testX[:, 0] - 3) * (testX[:, 0] - 3) / 2 + 1 - testX[:, 1] > 0

X = np.concatenate((trainX, testX), axis=0)
y = np.concatenate((trainy, testy), axis=0)
data = np.core.records.fromarrays([X[:, 0], X[:, 1], y], names='x1,x2,y')
df = pd.DataFrame(data)
df['timestamp'] = pd.date_range(start='1/1/2018', periods=len(df), freq='2H')
df.to_csv("case1.csv", index=None)

plt.plot(data[data['y']]['x1'], data[data['y']]['x2'], '+', color='r')
plt.plot(data[~data['y']]['x1'], data[~data['y']]['x2'], '+', color='b')
plt.show()
