"""
Sampler classes for use with BNP-Step and other PresseLab methods

Original version by J. Shep Bryan IV

"""

import math
import numpy as np
from scipy import stats
from scipy.special import gammaln
from scipy.special import gamma as gammafunc
from scipy.special import beta as betafunc

PI = np.pi
ln2 = np.log(2)
lnPI = np.log(PI)


class Normal:

    def __init__(self):
        pass

    @staticmethod
    def sample(mean, variance):
        if np.isscalar(mean * variance):
            return mean + np.sqrt(variance) * np.random.randn()
        else:
            return mean + np.sqrt(variance) * np.random.standard_normal((mean * variance).shape)

    @staticmethod
    def pdf(X, mean, variance):
        return (2 * PI * variance) ** (-.5) * np.exp(-.5 * (X - mean) ** 2 / variance)

    @staticmethod
    def logpdf(X, mean, variance):
        return -.5 * np.log(2 * PI * variance) - .5 * (X - mean) ** 2 / variance


class Gamma:

    def __init__(self):
        pass

    @staticmethod
    def sample(shape, scale):
        if np.isscalar(shape * scale):
            return scale * np.random.gamma(shape)
        else:
            return scale * np.random.standard_gamma(shape, size=(shape * scale).shape)

    @staticmethod
    def pdf(X, shape, scale):
        return X ** (shape - 1) * np.exp(-X / scale) / (gammafunc(shape) * scale ** shape)

    @staticmethod
    def logpdf(X, shape, scale):
        return (shape - 1) * np.log(X) - X / scale - gammaln(shape) - shape * np.log(scale)


class Beta:

    def __init__(self):
        pass

    @staticmethod
    def sample(self, a, b):
        return np.random.beta(a, b)

    @staticmethod
    def logpdf(self, X, a, b):
        prob = (
            (a - 1) * np.log(X)
            + (b - 1) * np.log(1 - X)
            + gammaln(a + b) - gammaln(a) - gammaln(b)
        )
        return prob


class Categorical:

    def __init__(self):
        pass

    @staticmethod
    def sample(p):
        idx = p > 0
        q = int(np.sum(np.random.rand()*np.sum(p[idx]) > np.cumsum(p[idx])))
        return np.where(idx)[0][q]

    @staticmethod
    def pdf(X, p):
        return p[int(X)] / np.sum(p)

    @staticmethod
    def logpdf(X, p):
        if np.any(p < 0):
            print('oh no!')
        value = np.log(p[int(X)])
        value -= np.log(np.sum(p))
        return value


class Binomial:

    def __init__(self):
        pass

    @staticmethod
    def sample(n, p):
        return np.random.binomial(n, p)

    @staticmethod
    def pdf(X, n, p):
        return math.factorial(n) // math.factorial(X) // math.factorial(n-X) * (p**X) * ((1-p)**(n-X))


class Exponential:

    def __init__(self):
        pass

    @staticmethod
    def sample(lam, loc=0):
        if np.isscalar(lam):
            return loc + np.random.exponential()/lam
        else:
            return loc + np.random.standard_exponential(size=lam.shape) / lam

    @staticmethod
    def pdf(X, lam, loc=0):
        return lam * np.exp(-lam * (X - loc))

    @staticmethod
    def logpdf(X, lam, loc=0):
        return np.log(lam) - lam * (X - loc)


class Dirichlet:

    def __init__(self):
        pass

    @staticmethod
    def sample(concentration):
        X = Gamma().sample(concentration, 1)
        if np.ndim(concentration) == 1:
            return X/np.sum(X)
        else:
            for k in range(X.shape[0]):
                X[k, :] /= np.sum(X[k, :])
            return X

    @staticmethod
    def logpdf(X, conc):

        if np.ndim(X) == 1:
            K = len(X)
            prob = gammaln(np.sum(conc))
            for k in range(K):
                if (X[k] > 0) and (conc[k]):
                    prob += (conc[k] - 1) * np.log(X[k]) - gammaln(conc[k])
        elif np.ndim(X) == 2:
            J, K = X.shape
            prob = 0
            for j in range(J):
                for k in range(K):
                    prob += gammaln(np.sum(conc[j, :]))
                    if (X[j, k] > 0) and (conc[j, k]):
                        prob += (conc[j, k] - 1) * np.log(X[j, k]) - gammaln(conc[j, k])

        return prob


class InvGamma:

    def __init__(self):
        pass

    @staticmethod
    def sample(shape, scale):
        if np.isscalar(shape * scale):
            return scale / np.random.gamma(shape)
        else:
            return scale / np.random.standard_gamma(shape, size=(shape * scale).shape)

    @staticmethod
    def logpdf(X, shape, scale):
        value = np.sum(
            shape * np.log(scale) - gammaln(shape) - (shape - 1) * np.log(X) - shape / X
        )
        return value


class MultivariateGaussian:

    def __init__(self):
        pass

    @staticmethod
    def sample(mu, sigma=None, sigma_chol=None, epsilon=0):

        num_data = len(mu)

        if np.isscalar(sigma):
            sigma_chol = np.sqrt(sigma)*np.eye(num_data)

        if sigma_chol is None:
            sigma_chol = np.linalg.cholesky(sigma + epsilon*np.eye(num_data))

        return np.array(mu).reshape((-1, 1)) + sigma_chol @ np.random.standard_normal(size=(num_data, 1))

    @staticmethod
    def logpdf(X, mu, sigma=None, sigma_inv=None, sigma_det=None, epsilon=0):

        if np.ndim(X) == 1:
            X = X.reshape((-1, 1))

        if np.ndim(mu) == 1:
            mu = mu.reshape((-1, 1))

        k = (X*mu).shape[0]

        if np.isscalar(sigma):
            sigma_inv = np.eye(k)/sigma

        if sigma_inv is None:
            sigma_inv = np.linalg.inv(sigma + epsilon*np.eye(k))

        if sigma_det is None:
            sigma_det = np.linalg.det(sigma + epsilon*np.eye(k))

        prob = - .5 * k * (ln2 + lnPI) - .5 * np.log(sigma_det) - .5 * ((X - mu).T @ sigma_inv @ (X - mu))
        prob = prob.reshape(1)[0]

        return prob









