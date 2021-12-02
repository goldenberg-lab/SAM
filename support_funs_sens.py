"""
THIS SCRIPT CONTAINS THE SUPPORT FUNCTIONS NEEDED TO CARRY OUT SAP
"""

# Load necessary modules
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, t
from sklearn.utils import resample
from time import time


def beta_fun(n, pt, pm, alpha):
    ta = stats.norm.ppf(1-alpha)
    sigma_t = np.sqrt(pt*(1-pt)/n)
    sigma_m = np.sqrt(pm*(1-pm)/n)
    Phi = stats.norm.cdf( (sigma_t*ta-(pm-pt))/sigma_m )
    return Phi

def perf_fun(*args, **kwargs):
    """
    Function to calculate the performance metric of interest
    1) You must use *args and **kwargs
    2) 'thresh' must be one of the kwargs
    3) This function must return a scalar
    """
    assert len(args) == 2
    assert 'thresh' in kwargs
    thresh = kwargs['thresh']
    y, score = args[0], args[1]
    assert np.all( (y==0) | (y==1) )
    assert thresh <= score.max()
    yhat = np.where(score >= thresh, 1, 0)
    sens = np.mean(yhat[y == 1])
    return sens

# args=(df.y.values, df.score.values);kwargs={'target':0.8}
def thresh_find(*args, **kwargs):
    """
    Function to find threshold for performance of interest
    1) You must use *args and **kwargs
    2) 'target' must be one of the kwargs. This is the value you want to get from perf_fun
    3) 'jackknife' must be an optional argument in kwargs that will return the function output by leaving one observation out
        See: https://en.wikipedia.org/wiki/Jackknife_resampling
        Note that many statistics have fast way to calculate the jackknife beyond brute-force
    4) This function must return a scalar, or a np.array is jackknife=True
    """
    # --- assign --- #
    jackknife = False
    ret_df = False
    if 'jackknife' in kwargs:
        jackknife = kwargs['jackknife']
    assert 'target' in kwargs
    target = kwargs['target']
    assert len(args) == 2
    y, score = args[0], args[1]
    assert len(y) == len(score)
    assert np.all((y==0) | (y==1))
    # --- Find quantile --- #
    s1 = np.sort(score[y == 1])
    n1 = len(s1)
    n0 = len(y) - n1
    sidx = np.arange(n1,0,-1) / n1
    sidx = np.argmax(np.where(sidx >= target)[0])
    tstar = np.quantile(s1, 1-target)
    if jackknife:
        # Effect of dropping an observation on the choice of the sensivity threshold
        tstar0 = np.repeat(tstar, n0)  # Zero class has no impact
        tmed = np.quantile(np.delete(s1,sidx),1-target)
        thigh = np.quantile(np.delete(s1,sidx-1),1-target)
        tlow = np.quantile(np.delete(s1,sidx+1),1-target)
        assert tlow <= tmed <= thigh
        tstar1 = np.append(np.repeat(thigh,sidx), np.array([tmed]))
        tstar1 = np.append(tstar1, np.repeat(tlow,n1-sidx-1))
        tstar = np.append(tstar0, tstar1)
    return tstar


def draw_samp(*args, strata=None):
    """
    FUNCTION DRAWS DATA WITH REPLACEMENT (WITH STRATIFICATION IF DESIRED)
    """
    args = list(args)
    if strata is not None:
        out = resample(*args, stratify=strata)
    else:
        out = resample(*args)
    if len(args) == 1:
        out = [out]
    return out


class bootstrap():
    def __init__(self, nboot, func):
        self.nboot = nboot
        self.stat = func
    
    def fit(self, *args, mm=100, **kwargs):
        strata=None
        if 'strata' in kwargs:
            strata = kwargs['strata']
        # Get the baseline stat
        self.theta = self.stat(*args, **kwargs)
        self.store_theta = np.zeros(self.nboot)
        self.jn = self.stat(*args, **kwargs, jackknife=True)
        self.n = len(self.jn)
        stime = time()
        for ii in range(self.nboot):  # Fit bootstrap
            if (ii+1) % mm == 0:
                nleft = self.nboot - (ii+1)
                rtime = time() - stime
                rate = (ii+1)/rtime
                eta = nleft / rate
                #print('Bootstrap %i of %i (ETA=%0.1f minutes)' % (ii+1, self.nboot, eta/60))
            args_til = draw_samp(*args, strata=strata)
            self.store_theta[ii] = self.stat(*args_til, **kwargs)
        self.se = self.store_theta.std()
                
    def get_ci(self, alpha=0.05, symmetric=True):
        assert (symmetric==True) | (symmetric=='upper') | (symmetric=='lower') 
        self.di_ci = {'quantile':[], 'se':[], 'bca':[]}
        self.di_ci['quantile'] = self.ci_quantile(alpha, symmetric)
        self.di_ci['se'] = self.ci_se(alpha, symmetric)
        self.di_ci['bca'] = self.ci_bca(alpha, symmetric)

    def ci_quantile(self, alpha, symmetric):
        if symmetric==True:
            return np.quantile(self.store_theta, [alpha/2,1-alpha/2])
        elif symmetric == 'lower':
            return np.quantile(self.store_theta, alpha)
        else:
            return np.quantile(self.store_theta, 1-alpha)
        
    def ci_se(self, alpha, symmetric):
        if symmetric==True:
            qq = t(df=self.n).ppf(1-alpha/2)
            return np.array([self.theta - self.se*qq, self.theta + self.se*qq])
        else:
            qq = t(df=self.n).ppf(1-alpha)
        if symmetric == 'lower':
            return self.theta - qq*self.se
        else:
            return self.theta + qq*self.se
    
    def ci_bca(self, alpha, symmetric):
        if symmetric==True:
            ql, qu = norm.ppf(alpha/2), norm.ppf(1-alpha/2)
        else:
            ql, qu = norm.ppf(alpha), norm.ppf(1-alpha)
        # Acceleration factor
        num = np.sum((self.jn.mean() - self.jn)**3)
        den = 6*np.sum((self.jn.mean() - self.jn)**2)**1.5
        self.ahat = num / den
        # Bias correction factor
        self.zhat = norm.ppf(np.mean(self.store_theta < self.theta))
        self.a1 = norm.cdf(self.zhat + (self.zhat + ql)/(1-self.ahat*(self.zhat+ql)))
        self.a2 = norm.cdf(self.zhat + (self.zhat + qu)/(1-self.ahat*(self.zhat+qu)))
        
        if symmetric==True:
            return np.quantile(self.store_theta, [self.a1, self.a2])
        elif symmetric=='lower':
            return np.quantile(self.store_theta, self.a1)
        else:
            return np.quantile(self.store_theta, self.a2)
        
        
        