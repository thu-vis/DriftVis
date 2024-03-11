import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# first 500 N((1,2,3), I)
# next 500 N((1,2,3), I) & N((3,2,4), I)
# next 500 N((1,1.5,2.5), I) & N((3,2,4), I)
# next 500 N((0.75,1.25,2.25), I) & N((3,2,4.25), I)
# next 500 N((0.75,1,2), I) & N((3,2,4.25), I)

np.random.seed(42)
n = 500
mix_rate = 0.6
n1 = int(n * mix_rate)
epsilon = 0.3

train1 = np.random.multivariate_normal(np.array([1, 2, 3]), np.identity(3), n)
train21 = np.random.multivariate_normal(np.array([1, 2, 3]), np.identity(3), n1)
train22 = np.random.multivariate_normal(np.array([3, 2, 5]), np.identity(3), n - n1)
train2 = np.concatenate([train21, train22])
np.random.shuffle(train2)
train31 = np.random.multivariate_normal(np.array([1, 1.5, 2.5]), np.identity(3), n1)
train32 = np.random.multivariate_normal(np.array([3, 2, 5]), np.identity(3), n - n1)
train3 = np.concatenate([train31, train32])
np.random.shuffle(train3)
train41 = np.random.multivariate_normal(np.array([0.75, 1.25, 2.25]), np.identity(3), n1)
train42 = np.random.multivariate_normal(np.array([3, 2, 5]), np.identity(3), n - n1)
train4 = np.concatenate([train41, train42])
np.random.shuffle(train4)
train51 = np.random.multivariate_normal(np.array([0.75, 1, 2]), np.identity(3), n1)
train52 = np.random.multivariate_normal(np.array([3, 2, 5]), np.identity(3), n - n1)
train5 = np.concatenate([train51, train52])
np.random.shuffle(train5)

trainX = np.concatenate((train1, train2, train3, train4, train5), axis=0)
border = np.random.uniform(-epsilon, epsilon, trainX.shape[0])
trainy = 4 + trainX[:, 0] * trainX[:, 0] + trainX[:, 1] * trainX[:, 1] - trainX[:, 2] * trainX[:, 2]> border

data = np.core.records.fromarrays([trainX[:, 0], trainX[:, 1], trainX[:, 2], trainy], names='x1,x2,x3,y')
df = pd.DataFrame(data)
df['timestamp'] = pd.date_range(start='1/1/2018', periods=len(df), freq='2H')
df.to_csv("case2.csv", index=None)


plt.plot(data[data['y']]['x1'], data[data['y']]['x3'], '+', color='r')
plt.plot(data[~data['y']]['x1'], data[~data['y']]['x3'], '+', color='b')
plt.show()

