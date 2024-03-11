import numpy as np
from scipy.stats.distributions import chi2, f


class Hypothesis(object):
    @staticmethod
    def WTest(x, sigma):
        '''
        test whether the sample covariance is identical to sigma
        :param x: a.shape == (d, n)
        :param sigma: the covariance matrix that the hypothesis holds
        :return: the P-value
        '''
        d, n = x.shape
        L = np.linalg.cholesky(sigma)
        y = np.linalg.pinv(L).dot(x)
        Sy = np.cov(y)
        W = np.trace((Sy - np.identity(d)).dot((Sy - np.identity(d)))) / d - np.trace(Sy) * np.trace(Sy) / n / d + d / n
        return chi2.sf(W * n * d / 2, d * (d + 1) / 2)

    @staticmethod
    def distribution_WTest(component, sigma):
        '''
        test whether the sample covariance is identical to sigma
        :param component: the gaussian component
        :param sigma: the covariance matrix that the hypothesis holds
        :return: the P-value
        '''
        d, n = component['mu'].shape[0], component['n']
        Linv = np.linalg.pinv(np.linalg.cholesky(sigma))
        Sy = Linv.dot(component['s']).dot(Linv.T)
        W = np.trace((Sy - np.identity(d)).dot((Sy - np.identity(d)))) / d - np.trace(Sy) * np.trace(Sy) / n / d + d / n
        return chi2.sf(W * n * d / 2, d * (d + 1) / 2)

    @staticmethod
    def HTest(x, mu):
        '''
        test whether the sample mean is identical to x
        :param x: a.shape == (d, n)
        :param mu: the mu matrix that the hypothesis holds
        :return: the P-value
        '''
        d, n = x.shape
        S = np.cov(x)
        xbar_mu = np.mean(x, axis=1) - mu
        T2 = xbar_mu.dot(np.linalg.pinv(S)).dot(xbar_mu) * n
        return f.sf((n - d) * T2 / d / (n - 1), d, n - d)

    @staticmethod
    def distribution_HTest(component, mu):
        '''
        test whether the sample covariance is identical to sigma
        :param component: the gaussian component
        :param sigma: the covariance matrix that the hypothesis holds
        :return: the P-value
        '''
        d, n = component['mu'].shape[0], component['n']
        xbar_mu = component['mu'] - mu
        T2 = xbar_mu.dot(component['sinv']).dot(xbar_mu) * n
        return f.sf((n - d) * T2 / d / (n - 1), d, n - d)