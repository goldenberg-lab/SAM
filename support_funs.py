"""
THIS SCRIPT CONTAINS THE SUPPORT FUNCTIONS NEEDED TO CARRY OUT SAP
"""

# Load necessary modules
import numpy as np
import pandas as pd
from scipy.stats import norm, t
from sklearn.utils import resample


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
    num_tp = np.sum(yhat[y == 1] == 1)
    num_fp = np.sum(yhat[y == 0] == 1)
    ppv = num_tp / (num_tp + num_fp)
    return ppv


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
    if 'ret_df' in kwargs:
        ret_df = kwargs['ret_df']
    assert 'target' in kwargs
    target = kwargs['target']
    assert len(args) == 2
    y, score = args[0], args[1]
    assert len(y) == len(score)
    assert np.all((y==0) | (y==1))
    # --- calculate --- #
    s0, s1 = score[y == 0], score[y == 1]
    u_scores = np.sort(score)  # Useful for step function
    store = np.zeros([len(u_scores),2],int)
    for ii, tt in enumerate(u_scores):
        store[ii] = [np.sum(s0 >= tt), np.sum(s1 >= tt)]
    dat = pd.DataFrame(store,columns=['n0','n1']).assign(thresh=u_scores,tot=store.sum(1))
    dat = dat.assign(thresh1=lambda x: x.thresh.shift(1), ppv=lambda x: x.n1/(x.tot))
    dat = dat.assign(ppv1=lambda x: x.ppv.shift(1), tot1=lambda x: x.tot.shift(1)).iloc[1:]
    if ret_df:
        return dat
    tstar = thresh_interp(dat, target)
    # Do a fast interpolation with the Jackknife
    # Remember: all s[1]<t and s[0]<t do not impact calculation (i.e. False negatives and True Negatives)
    if jackknife:
        tmp = dat.query('thresh>=@tstar & thresh1<@tstar')
        n0, n1, tot0, tot1 = tmp.n0.values[0], tmp.n1.values[0], tmp.tot1.values[0], tmp.tot.values[0]
        thresh0, thresh1 = tmp.thresh1.values[0], tmp.thresh.values[0]
        ppv0, ppv1 = tmp.ppv1.values[0], tmp.ppv.values[0]
        holder = []
        holder.append(np.repeat(tstar,len(score) - tot1)) # Removing all false/true negatives
        # Slope for removing TP
        ppv1_new, ppv0_new = (n1-1)/tot1, (n1-1)/tot0
        slope_new = (ppv1_new - ppv0_new) / (thresh1 - thresh0)
        assert ppv1_new < ppv1  # Has to decrease
        holder.append(np.repeat(thresh1 + (ppv1 - ppv1_new)/slope_new, n1))
        # Note that becasue n1/(tot0-1) = n1/tot1, implies thresh0 will be be the new choice
        holder.append(np.repeat(thresh0, n0))
        tstar = np.concatenate(holder)
        tstar = tstar[np.abs(tstar)!=np.Inf]
    return tstar

        
def thresh_interp(df, target):
    """
    LINEARLY INTERPOLATES PPV TO FIND THRESHOLD
    """
    df = df.assign(err=lambda x: x.ppv - target).assign(err1 = lambda x: x.ppv1 - target)
    idx = df.ppv.isnull()
    if idx.sum() > 0:
        df = df[~idx]
    if df.ppv.max() < target:
        #print('exceeds max')
        df = df.query('ppv == ppv.max()').sort_values('thresh1').head(1)
    elif df.ppv.min() > target:
        #print('less than max')
        df = df.query('ppv == ppv.min()').sort_values('thresh1').head(1)
    else:
        df = df[((np.sign(df.err1)==-1) & (np.sign(df.err)==1)) | 
                ((np.sign(df.err1)==-1) & (np.sign(df.err)==0))]
        df = df.sort_values('thresh1').head(1)
    thresh0, thresh1 = df.thresh1.values[0], df.thresh.values[0]
    ppv0, ppv1 = df.ppv1.values[0], df.ppv.values[0]
    slope = (ppv1 - ppv0) / (thresh1 - thresh0)
    tt = thresh1 - (ppv1 - target)/slope
    return tt


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
    
    def fit(self, *args, **kwargs):
        strata=None
        if 'strata' in kwargs:
            strata = kwargs['strata']
        # Get the baseline stat
        self.theta = self.stat(*args, **kwargs)
        self.store_theta = np.zeros(self.nboot)
        self.jn = self.stat(*args, **kwargs, jackknife=True)
        self.n = len(self.jn)
        for ii in range(self.nboot):  # Fit bootstrap
            #if (ii+1) % 1000 == 0:
                #print('Bootstrap %i of %i' % (ii+1, self.nboot))
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
        
        
        