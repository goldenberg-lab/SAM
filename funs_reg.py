import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from scipy.stats import multivariate_normal as MVN

class cond_dist():
    def __init__(self, k, n1, n2, null=False):
        self.s = +1
        if null is True:
            self.s = -1
        self.k, self.n1, self.n2 = k, n1, n2
        self.r = np.sqrt(n2 / n1)
    
    def cdf_w(self, w):
        a = np.sqrt(1+self.r**2) * self.k * self.s
        b = -self.r * self.s
        rho = -b/np.sqrt(1+b**2)
        Sigma = np.array([[1,rho],[rho,1]])
        dist_MVN = MVN(mean=np.repeat(0,2),cov=Sigma)
        x1 = a / np.sqrt(1+b**2)
        if isinstance(w, float):
            X = [x1, w]
        else:
            X = np.c_[np.repeat(x1,len(w)), w]
        pval = dist_MVN.cdf(X)
        return pval
    
    def cdf_x(self, x):
        const = 1 / norm.cdf(self.s * self.k)
        w = (x + self.r * self.k) / np.sqrt(1+self.r**2)
        pval = self.cdf_w(w) * const
        return pval
    
    def quantile(self, p):
        res = minimize_scalar(fun=lambda x: (self.cdf_x(x)-p)**2, method='brent').x
        return res


def power_est(n2, k, n1, alpha):
    dist_true = cond_dist(k=k, n1=n1, n2=n2, null=True)
    dist_false = cond_dist(k=k, n1=n1, n2=n2, null=False)
    crit_value = dist_true.quantile(alpha)
    power = dist_false.cdf_x(crit_value)
    return power

def power_find(pp, k, n1, alpha):
    n2 = minimize_scalar(fun=lambda x: (power_est(x, k, n1, alpha)-pp)**2,method='brent').x
    return n2


