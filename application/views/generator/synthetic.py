import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

'''
data: x1, x2
[0,500): distribution 1
[500,1000) distribution 1 + distribution 2
[1000,1500) distribution 2
[1500,2000) distribution 3
[2000,2500) distribution 1 + distribution 2 + distribution 3
'''


np.random.seed(42)
n = 500
epsilon = 0.5

mean_1, cov_1 = np.array([-2, 4]), np.array([[1, 0], [0, 1]])
mean_2, cov_2 = np.array([0, 0]), np.array([[1, 0], [0, 1]])
mean_3, cov_3 = np.array([2, 4]), np.array([[1, 0], [0, 1]])

train1 = np.random.multivariate_normal(mean_1, cov_1, n)
train2_1 = np.random.multivariate_normal(mean_1, cov_1, n // 2)
train2_2 = np.random.multivariate_normal(mean_2, cov_2, n // 2)
train2 = np.zeros((n, 2))
tmp = [90, 70, 50, 30, 10]
_i = 0
_j = 0
_k = 0
for _n in tmp:
    train2[_k:_k + _n, :] = train2_1[_i:_i + _n, :]
    _i += _n
    _k += _n
    train2[_k:_k + 100 - _n, :] = train2_2[_j:_j + 100 - _n, :]
    _j += 100 - _n
    _k += 100 - _n
    np.random.shuffle(train2[_k-100:_k])
train3 = np.random.multivariate_normal(mean_2, cov_2, n)
train4 = np.random.multivariate_normal(mean_3, cov_3, n)
train5_1 = np.random.multivariate_normal(mean_1, cov_1, n // 3)
train5_2 = np.random.multivariate_normal(mean_2, cov_2, n // 3)
train5_3 = np.random.multivariate_normal(mean_3, cov_3, n - n // 3 - n // 3)
train5 = np.concatenate([train5_1, train5_2, train5_3])

X = np.concatenate((train1, train2, train3, train4, train5))
border = np.random.uniform(-epsilon, epsilon, X.shape[0])
border[2000:] = 0
y = X[:, 0] * X[:, 0] - X[:, 1] > border
y_2 = y.astype(int)
y_2[X[:, 0] < -2+border] = 2
y_2[X[:, 0] > 2-border] = 2
y_val = np.sin(X[:,0] * X[:,0])+np.cos(X[:,1] * X[:,1]) + np.concatenate([np.random.normal(0,0.1,2000),np.zeros(500)])


data = np.core.records.fromarrays([X[:, 0], X[:, 1], y, y_2, y_val], names='x1,x2,y,y_2,y_val')
df = pd.DataFrame(data)
df.to_csv("synthetic.csv", index=None)

plt.plot(data[data['y']]['x1'], data[data['y']]['x2'], '+', color='r')
plt.plot(data[~data['y']]['x1'], data[~data['y']]['x2'], '+', color='b')
plt.show()
plt.close()

plt.plot(data[data['y_2']==0]['x1'], data[data['y_2']==0]['x2'], '+', color='r')
plt.plot(data[data['y_2']==1]['x1'], data[data['y_2']==1]['x2'], '+', color='g')
plt.plot(data[data['y_2']==2]['x1'], data[data['y_2']==2]['x2'], '+', color='b')
plt.show()
plt.close()
