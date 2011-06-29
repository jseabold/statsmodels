#Kaplan-Meier Estimator

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy import stats

from statsmodels.iolib.table import SimpleTable
from statsmodels.base.model import LikelihoodModel, LikelihoodModelResults
import itertools

##Need to update all docstrings
##Use assume unique for np.in1d?

class Survival(object):

    ##Add handling for non-integer times

    """
    Survival(...)
        Survival(data, time1, time2=None, censoring=None)

        Create an object to store survival data for precessing
        by other survival analysis functions

        Parameters
        -----------

        censoring: index of the column containing an indicator
            of whether an observation is an event, or a censored
            observation, with 0 for censored, and 1 for an event

        data: array_like
            An array, with observations in each row, and
            variables in the columns

        time1 : if time2=None, index of comlumn containing the duration
            that the suject survivals and remains uncensored (e.g. observed
            survival time), if time2 is not None, then time1 is the index of
            a column containing start times for the observation of each subject
            (e.g. oberved survival time is end time minus start time)

        time2: index of column containing end times for each observation

        Attributes
        -----------

        times: vectore of survival times

        censoring: vector of censoring indicators

        Examples
        ---------

        see other survival analysis functions for examples of usage with those
        functions

    """

    def __init__(self, time1, time2=None, censoring=None, data=None):
        if not data is None:
            data = np.asarray(data)
            if censoring is None:
                self.censoring = None
            else:
                self.censoring = (data[:,censoring]).astype(int)
            if time2 is None:
                self.times = (data[:,time1]).astype(int)
            else:
                self.times = (((data[:,time2]).astype(int))
                              - ((data[:,time1]).astype(int)))
        else:
            time1 = (np.asarray(time1)).astype(int)
            if not time2 is None:
                time2 = (np.array(time2)).astype(int)
                self.times = time2 - time1
            else:
                self.times = time1
            if censoring is None:
                self.censoring == None
            else:
                self.censoring = (np.asarray(censoring)).astype(int)


class KaplanMeier(object):

    """
    KaplanMeier(...)
        KaplanMeier(data, endog, exog=None, censoring=None)

        Create an object of class KaplanMeier for estimating
        Kaplan-Meier survival curves.

        Parameters
        ----------
        data: array_like
            An array, with observations in each row, and
            variables in the columns

        surv: Survival object containing desire times and censoring

        endog: index (starting at zero) of the column
            containing the endogenous variable (time)

        exog: index of the column containing the exogenous
            variable (must be catagorical). If exog = None, this
            is equivalent to a single survival curve. Alternatively,
            this can be a vector of exogenous variables index in the same
            manner as data provided either from data or surv

        censoring: index of the column containing an indicator
            of whether an observation is an event, or a censored
            observation, with 0 for censored, and 1 for an event


        Attributes
        -----------
        censorings: List of censorings associated with each unique
            time, at each value of exog

        events: List of the number of events at each unique time
            for each value of exog

        results: List of arrays containing estimates of the value
            value of the survival function and its standard error
            at each unique time, for each value of exog

        ts: List of unique times for each value of exog

        Methods
        -------
        fit: Calcuate the Kaplan-Meier estimates of the survival
            function and its standard error at each time, for each
            value of exog

        plot: Plot the survival curves using matplotlib.plyplot

        summary: Display the results of fit in a table. Gives results
            for all (including censored) times

        test_diff: Test for difference between survival curves

        Examples
        --------
        >>> import statsmodels.api as sm
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from statsmodels.sandbox.survival2 import KaplanMeier
        >>> dta = sm.datasets.strikes.load()
        >>> dta = dta.values()[-1]
        >>> dta[range(5),:]
        array([[  7.00000000e+00,   1.13800000e-02],
               [  9.00000000e+00,   1.13800000e-02],
               [  1.30000000e+01,   1.13800000e-02],
               [  1.40000000e+01,   1.13800000e-02],
               [  2.60000000e+01,   1.13800000e-02]])
        >>> km = KaplanMeier(dta,0)
        >>> km.fit()
        >>> km.plot()

        Doing

        >>> km.summary()

        will display a table of the estimated survival and standard errors
        for each time. The first few lines are

                  Kaplan-Meier Curve
        =====================================
         Time     Survival        Std. Err
        -------------------------------------
         1.0   0.983870967742 0.0159984306572
         2.0   0.91935483871  0.0345807888235
         3.0   0.854838709677 0.0447374942184
         4.0   0.838709677419 0.0467104592871
         5.0   0.822580645161 0.0485169952543

        Doing

        >>> plt.show()

        will plot the survival curve

        Mutliple survival curves:

        >>> km2 = KaplanMeier(dta,0,exog=1)
        >>> km2.fit()

        km2 will estimate a survival curve for each value of industrial
        production, the column of dta with index one (1).

        With censoring:

        >>> censoring = np.ones_like(dta[:,0])
        >>> censoring[dta[:,0] > 80] = 0
        >>> dta = np.c_[dta,censoring]
        >>> dta[range(5),:]
        array([[  7.00000000e+00,   1.13800000e-02,   1.00000000e+00],
               [  9.00000000e+00,   1.13800000e-02,   1.00000000e+00],
               [  1.30000000e+01,   1.13800000e-02,   1.00000000e+00],
               [  1.40000000e+01,   1.13800000e-02,   1.00000000e+00],
               [  2.60000000e+01,   1.13800000e-02,   1.00000000e+00]])

        >>> km3 = KaplanMeier(dta,0,exog=1,censoring=2)
        >>> km3.fit()

        Test for difference of survival curves

        >>> log_rank = km3.test_diff([0.0645,-0.03957])

        The zeroth element of log_rank is the chi-square test statistic
        for the difference between the survival curves for exog = 0.0645
        and exog = -0.03957, the index one element is the degrees of freedom for
        the test, and the index two element is the p-value for the test

        Groups with nan names

        >>> groups = np.ones_like(dta[:,1])
        >>> groups = groups.astype('S4')
        >>> groups[dta[:,1] > 0] = 'high'
        >>> groups[dta[:,1] <= 0] = 'low'
        >>> dta = dta.astype('S4')
        >>> dta[:,1] = groups
        >>> dta[range(5),:]
        array([['7.0', 'high', '1.0'],
               ['9.0', 'high', '1.0'],
               ['13.0', 'high', '1.0'],
               ['14.0', 'high', '1.0'],
               ['26.0', 'high', '1.0']],
              dtype='|S4')
        >>> km4 = KaplanMeier(dta,0,exog=1,censoring=2)
        >>> km4.fit()

    """

    ##Add stratification

    def __init__(self, surv, exog=None, data=None):
        censoring = surv.censoring
        times = surv.times
        if exog is not None:
            if data is not None:
                data = np.asarray(data)
                if data.ndim != 2:
                    raise ValueError("Data array must be 2d")
                exog = data[:,exog]
            else:
                exog = np.asarray(exog)
        if exog is None:
            self.exog = None
            if censoring != None:
                data = np.c_[times,censoring]
                data = data[~np.isnan(data).any(1)]
                self.times = (data[:,0]).astype(int)
                self.censoring = (data[:,1]).astype(int)
                del(data)
            else:
                self.times = times[~np.isnan(times)]
        elif exog.dtype == float or exog.dtype == int:
            if censoring != None:
                data = np.c_[times,censoring,exog]
                data = data[~np.isnan(data).any(1)]
                self.times = (data[:,0]).astype(int)
                self.censoring = (data[:,1]).astype(int)
                self.exog = data[:,2:]
            else:
                data = np.c_[times,exog]
                data = data[~np.isnan(data).any(1)]
                self.times = (data[:,0]).astype(int)
                self.exog = data[:,1:]
            del(data)
        else:
            exog = exog[~np.isnan(times)]
            if censoring is not None:
                censoring = censoring[~np.isnan(times)]
            times = times[~np.isnan(times)]
            if censoring is not None:
                times = (times[~np.isnan(censoring)]).astype(int)
                exog = exog[~np.isnan(censoring)]
                censoring = (censoring[~np.isnan(censoring)]).astype(int)
            if exog.ndim == 2:
                self.times = (times[~np.isnan(exog).any(1)]).astype(int)
                self.censoring = (censoring[~np.isnan(exog).any(1)]).astype(int)
                self.exog = (exog[~np.isnan(exog).any(1)]).astype(float)
            else:
                self.times = (times[~np.isnan(exog)]).astype(int)
                self.censoring = (censoring[~np.isnan(exog)]).astype(int)
                self.exog = (exog[~np.isnan(exog)]).astype(float)
        if exog is not None:
            if self.exog.ndim == 2 and len(self.exog[0]) == 1:
                self.exog = self.exog[:,0]
            self.df_resid = len(exog) - 1
        else:
            self.df_resid = 1

    def fit(self, CI_transform="log-log", force_CI_0_1=True):
        """
        Calculate the Kaplan-Meier estimator of the survival function
        """
        exog = self.exog
        censoring = self.censoring
        times = self.times
        self.results = []
        self.ts = []
        self.censorings = []
        self.event = []
        self.params = np.array([])
        self.normalized_cov_params = np.array([])
        if exog is None:
            self.groups = None
            self._fitting_proc(times, censoring, CI_transform,
                              force_CI_0_1)
        else:
            ##Can remove second part of condition?
            if exog.ndim == 2:
                groups = stats._support.unique(exog)
                self.groups = groups
                ##ncols = len(exog[0])
                ##groups = np.unique(exog)
                ##groups = np.repeat(groups, ncols)
                ##need different iterator for rows with repeats?
                ##groups = itertools.permutations(groups, ncols)
                ##groups = [i for i in groups]
                ##groups = np.array(groups)
                ##self.groups = 1
                for g in groups:
                    ##stats.adm for testing?
                    ind = np.product(exog == g, axis=1) == 1
                    if ind.any():
                        t = times[ind]
                        if censoring is not None:
                            c = censoring[ind]
                        else:
                            c = None
                        self._fitting_proc(t, c, CI_transform, force_CI_0_1)
                        ##if self.groups is 1:
                            ##self.groups = g
                        ##else:
                            ##self.groups = np.c_[self.groups, g]
                ##self.groups = self.groups.T
            else:
                groups = np.unique(self.exog)
                self.groups = groups
                for g in groups:
                    t = (times[exog == g])
                    if not censoring is None:
                        c = (censoring[exog == g])
                    else:
                        c = None
                    self._fitting_proc(t, c, CI_transform, force_CI_0_1)
        return KMResults(self, self.params, self.normalized_cov_params)

    def _fitting_proc(self, t, censoring, CI_transform, force_CI):
        """
        For internal use
        """
        if censoring is None:
            n = len(t)
            events = np.bincount(t)
            t = np.unique(t)
            events = events[:,list(t)]
            events = events.astype(float)
            eventsSum = np.cumsum(events)
            eventsSum = np.r_[0,eventsSum]
            n -= eventsSum[:-1]
        else:
            reverseCensoring = -1*(censoring - 1)
            events = np.bincount(t,censoring)
            censored = np.bincount(t,reverseCensoring)
            t = np.unique(t)
            censored = censored[:,list(t)]
            censored = censored.astype(float)
            censoredSum = np.cumsum(censored)
            censoredSum = np.r_[0,censoredSum]
            events = events[:,list(t)]
            events = events.astype(float)
            eventsSum = np.cumsum(events)
            eventsSum = np.r_[0,eventsSum]
            n = len(censoring) - eventsSum[:-1] - censoredSum[:-1]
            (self.censorings).append(censored)
        survival = np.cumprod(1-events/n)
        var = ((survival*survival) *
               np.cumsum(events/(n*(n-events))))
        se = np.sqrt(var)
        if CI_transform == "log":
            lower = (np.exp(np.log(survival) - 1.96 *
                                    (se * (1/(survival)))))
            upper = (np.exp(np.log(survival) + 1.96 *
                                    (se * (1/(survival)))))
        if CI_transform == "log-log":
            lower = (np.exp(-np.exp(np.log(-np.log(survival)) - 1.96 *
                                    (se * (1/(survival * np.log(survival)))))))
            upper = (np.exp(-np.exp(np.log(-np.log(survival)) + 1.96 *
                                    (se * (1/(survival * np.log(survival)))))))
        if force_CI:
            lower[lower < 0] = 0
            upper[upper > 1] = 1
        self.params = np.r_[self.params,survival]
        self.normalized_cov_params = np.r_[self.normalized_cov_params, se]
        (self.results).append(np.array([survival,se,lower,upper]))
        (self.ts).append(t)
        (self.event).append(events)

class CoxPH(LikelihoodModel):

    ##Add efron fitting, and other methods
    ##Add stratification

    """
    Fit a cox proportional harzard model from survival data
    """

    def __init__(self, surv, exog, data=None, ties="efron", strata=None):
        self.surv = surv
        self.ties = ties
        censoring = surv.censoring
        times = surv.times
        if data is not None:
            data = np.asarray(data)
            if data.ndim != 2:
                raise ValueError("Data array must be 2d")
            exog = data[:,exog]
        else:
            exog = np.asarray(exog)
        if exog.dtype == float or exog.dtype == int:
            if censoring != None:
                data = np.c_[times,censoring,exog]
                data = data[~np.isnan(data).any(1)]
                self.times = (data[:,0]).astype(int)
                self.censoring = (data[:,1]).astype(int)
                self.exog = data[:,2:]
            else:
                data = np.c_[times,exog]
                data = data[~np.isnan(data).any(1)]
                self.times = (data[:,0]).astype(int)
                self.exog = data[:,1:]
            del(data)
        else:
            exog = exog[~np.isnan(times)]
            if censoring is not None:
                censoring = censoring[~np.isnan(times)]
            times = times[~np.isnan(times)]
            if censoring is not None:
                times = (times[~np.isnan(censoring)]).astype(int)
                exog = exog[~np.isnan(censoring)]
                censoring = (censoring[~np.isnan(censoring)]).astype(int)
            if exog.ndim == 2:
                self.times = (times[~np.isnan(exog).any(1)]).astype(int)
                self.censoring = (censoring[~np.isnan(exog).any(1)]).astype(int)
                self.exog = (exog[~np.isnan(exog).any(1)]).astype(float)
            else:
                self.times = (times[~np.isnan(exog)]).astype(int)
                self.censoring = (censoring[~np.isnan(exog)]).astype(int)
                self.exog = (exog[~np.isnan(exog)]).astype(float)
        if strata is not None:
            self.stratify(strata, copy=False)
        else:
            self.strata = None
        ##Not need for stratification?
        ##List of ds for stratification?
        times = self.times
        d = np.bincount(times,self.censoring)
        times = np.unique(times)
        d = d[:,list(times)]
        self.d = (np.c_[times, d]).astype(float)
        self.df_resid = len(self.exog) - 1
        self.confint_dist = stats.norm

    def stratify(self, stratas, copy=True):

        """
        Create a CoxPH object to fit a model with stratification

        Parameters
        ----------
        strata: list of indicies columns of the matrix of exogenous variables
        that are to be included as strata. All other columns will be included
        as unstratified variables

        copy: logical value indicating whether a new CoxPH object sould be
        returned, or if the current object should be overwritten
        """

        exog = self.exog
        strata = exog[:,stratas]
        #exog = exog.compress(stratas, axis=1)
        if strata.ndim == 1:
            groups = np.unique(strata)
        elif strata.ndim == 2:
            groups = stats._support.unique(strata)
        if copy:
            model = CoxPH(self.surv, exog, ties=self.ties, strata=stratas)
            ##redundent in some cases?
            #model.exog = exog.compress(stratas, axis=1)
            #model.strata_groups = groups
            #model.strata = strata
            return model
        else:
            self.strata_groups = groups
            self.strata = strata
            self.exog = exog.compress(stratas, axis=1)

    def loglike(self, b):

        """
        Calculate the value of the log-likelihood at estimates of the
        parameters

        Parameters:
        ------------

        b: vector of parameter estimates

        Returns
        -------

        value of log-likelihood as a float
        """

        exog = self.exog
        times = self.times
        censoring = self.censoring
        d = self.d
        if self.strata is None:
            self._str_exog = exog
            self._str_times = times
            self._str_d = d
            if censoring is not None:
                self._str_censoring = censoring
            return self._loglike_proc(b)
        else:
            logL = 0
            for g in self.strata_groups:
                ind = np.product(self.strata == g, axis=1) == 1
                self._str_exog = exog[ind]
                _str_times = times[ind]
                self._str_times = _str_times
                if censoring is not None:
                    _str_censoring = censoring[ind]
                    self._str_censoring = _str_censoring
                ds = np.bincount(_str_times,_str_censoring)
                _str_times = np.unique(_str_times)
                ds = ds[:,list(_str_times)]
                self._str_d = (np.c_[_str_times, ds]).astype(float)
                #self._str_d = d[np.in1d(d[:,0], _str_times)]
                logL += self._loglike_proc(b)
            return logL

    def _loglike_proc(self, b):

        """
        Calculate the value of the log-likelihood at estimates of the
        parameters for each strata

        Parameters:
        ------------

        b: vector of parameter estimates

        Returns
        -------
        
        value of log-likelihood for strata as a float

        """

        ties = self.ties
        exog = self._str_exog
        times = self._str_times
        censoring = self._str_censoring
        d = self._str_d
        BX = np.dot(exog, b)
        thetas = np.exp(BX)
        d = d[d[:,1] != 0]
        c_idx = censoring == 1
        if ties == "efron":
            logL = 0
            for t in d[:,0]:
                ind = (c_idx) * (times == t)
                tied = d[d[:,0] == t][0][1]
                logL += ((np.dot(exog[ind], b)).sum()
                         - (np.log((thetas[times >= t]).sum()
                                   - ((np.arange(tied))/tied)
                                   * (thetas[ind]).sum()).sum()))
        if ties == "breslow":
            logL = (BX[c_idx]).sum()
            for t in d[:,0]:
                logL -= ((np.log((thetas[times >= t]).sum()))
                         * d[d[:,0] == t][0][1])
        return logL

    def score(self, b):

        """
        Calculate the score vector of the log-likelihood at estimates of the
        parameters

        Parameters:
        ------------

        b: vector of parameter estimates

        Returns
        -------

        value of score as 1d array
        """

        ties = self.ties
        exog = self.exog
        times = self.times
        censoring = self.censoring
        d = self.d
        BX = np.dot(exog, b)
        thetas = np.exp(BX)
        d = d[d[:,1] != 0]
        c_idx = censoring == 1
        if ties == "efron":
            score = 0
            for t in d[:,0]:
                ind = (c_idx) * (times == t)
                ind2 = times >= t
                thetaj = thetas[ind2]
                Xj = exog[ind2]
                thetai = thetas[ind]
                Xi = exog[ind]
                tied = d[d[:,0] == t][0][1]
                num1 = np.dot(thetaj, Xj)
                num2 = np.dot(thetai, Xi)
                de1 = thetaj.sum()
                de2 = thetai.sum()
                score += Xi.sum(0)
                for i in range(int(tied)):
                    c = i/tied
                    score -= (num1 - c * num2) / (de1 - c * de2)
        elif ties == "breslow":
            score = (exog[c_idx]).sum(0)
            for t in d[:,0]:
                ind = times >= t
                thetaj = thetas[ind]
                Xj = exog[ind]
                if ties == "breslow":
                    score -= ((np.dot(thetaj, Xj))/(thetaj.sum()) *
                              d[d[:,0] == t][0][1])
        return score

    def hessian(self, b):

        """
        Calculate the hessian matrix of the log-likelihoos at estimates of the
        parameters

        Parameters:
        ------------

        b: vector of parameter estimates

        Returns
        -------

        value of hessian as 2d array
        """

        ties = self.ties
        exog = self.exog
        times = self.times
        censoring = self.censoring
        d = self.d
        BX = np.dot(exog, b)
        thetas = np.exp(BX)
        d = d[d[:,1] != 0]
        hess = 0

        if ties == "efron":
            c_idx = censoring == 1
            for t in d[:,0]:
                ind = (c_idx) * (times == t)
                ind2 = times >= t
                thetaj = thetas[ind2]
                Xj = exog[ind2]
                thetai = thetas[ind]
                Xi = exog[ind]
                thetaXj = np.dot(thetaj, Xj)
                thetaXi = np.dot(thetai, Xi)
                tied = d[d[:,0] == t][0][1]
                num1 = np.dot(Xj.T, (Xj * thetaj[:,np.newaxis]))
                num2 = np.dot(Xi.T, (Xi * thetai[:,np.newaxis]))
                de1 = thetaj.sum()
                de2 = thetai.sum()
                for i in range(int(tied)):
                    c = i/tied
                    num3 = (thetaXj - c * thetaXi)
                    de = de1 - c * de2
                    hess += (((num1 - c * num2) / (de)) -
                             (np.dot(num3[:,np.newaxis], num3[np.newaxis,:])
                              / (de**2)))
        elif ties == "breslow":
            for t in d[:,0]:
                ind = times >= t
                thetaj = thetas[ind]
                Xj = exog[ind]
                thetaX = np.mat(np.dot(thetaj, Xj))
                ##Save more variables to avoid recalulation?
                hess += ((((np.dot(Xj.T, (Xj * thetaj[:,np.newaxis])))/(thetaj.sum()))
                         - ((np.array(thetaX.T * thetaX))/((thetaj.sum())**2))) *
                         d[d[:,0] == t][0][1])
        return -hess

    def information(self, b):
        return -self.hessian(b)

    def covariance(self, b):
        return la.pinv(self.information(b))

    def fit(self, start_params=None, method='newton', maxiter=100,
            full_output=1,disp=1, fargs=(), callback=None, retall=0, **kwargs):
        if start_params is None:
            self.start_params = np.zeros_like(self.exog[0])
        else:
            self.start_params = start_params
        results = super(CoxPH, self).fit(start_params, method, maxiter,
            full_output,disp, fargs, callback, retall, **kwargs)
        return CoxResults(self, results.params,
                               self.covariance(results.params))

class KMResults(LikelihoodModelResults):
    """
    Results for a Kaplan-Meier model
    """

    ##Add handling for stratification

    def __init__(self, model, params, normalized_cov_params=None, scale=1.0):
        super(KMResults, self).__init__(model, params, normalized_cov_params,
                                        scale)
        self.results = model.results
        self.times = model.times
        self.ts = model.ts
        self.censoring = model.censoring
        self.censorings = model.censorings
        self.exog = model.exog
        self.event = model.event
        self.groups = model.groups

    def test_diff(self, groups, rho=None, weight=None):

        """
        test_diff(groups, rho=0)

        Test for difference between survival curves

        Parameters
        ----------
        groups: A list of the values for exog to test for difference.
        tests the null hypothesis that the survival curves for all
        values of exog in groups are equal

        rho: compute the test statistic with weight S(t)^rho, where
        S(t) is the pooled estimate for the Kaplan-Meier survival function.
        If rho = 0, this is the logrank test, if rho = 0, this is the
        Peto and Peto modification to the Gehan-Wilcoxon test.

        weight: User specified function that accepts as its sole arguement
        an array of times, and returns an array of weights for each time
        to be used in the test

        Returns
        -------
        An array whose zeroth element is the chi-square test statistic for
        the global null hypothesis, that all survival curves are equal,
        the index one element is degrees of freedom for the test, and the
        index two element is the p-value for the test.

        Examples
        --------

        >>> import statsmodels.api as sm
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from statsmodels.sandbox.survival2 import KaplanMeier
        >>> dta = sm.datasets.strikes.load()
        >>> dta = dta.values()[-1]
        >>> censoring = np.ones_like(dta[:,0])
        >>> censoring[dta[:,0] > 80] = 0
        >>> dta = np.c_[dta,censoring]
        >>> km = KaplanMeier(dta,0,exog=1,censoring=2)
        >>> km.fit()

        Test for difference of survival curves

        >>> log_rank = km3.test_diff([0.0645,-0.03957])

        The zeroth element of log_rank is the chi-square test statistic
        for the difference between the survival curves using the log rank test
        for exog = 0.0645 and exog = -0.03957, the index one element
        is the degrees of freedom for the test, and the index two element
        is the p-value for the test

        >>> wilcoxon = km.test_diff([0.0645,-0.03957], rho=1)

        wilcoxon is the equivalent information as log_rank, but for the
        Peto and Peto modification to the Gehan-Wilcoxon test.

        User specified weight functions

        >>> log_rank = km3.test_diff([0.0645,-0.03957], weight=np.ones_like)

        This is equivalent to the log rank test

        More than two groups

        >>> log_rank = km.test_diff([0.0645,-0.03957,0.01138])

        The test can be performed with arbitrarily many groups, so long as
        they are all in the column exog
        """

        ##Fix with strata

        groups = np.asarray(groups)
        exog = self.exog
        pooled = self.groups
        if exog is None:
            raise ValueError("Need an exogenous variable for tests")

        elif (np.in1d(groups,self.groups)).all():
            if pooled.ndim == 1:
                ind = np.in1d(exog,groups)
                t = self.times[ind]
            else:
                ind = 0
                for g in groups:
                    ##More elegant method, append times?
                    ind += np.product(exog == g, axis=1)
                t = self.times[ind > 0]
            if not self.censoring is None:
                censoring = self.censoring[ind]
            else:
                censoring = None
            del(ind)
            tind = np.unique(t)
            NK = []
            N = []
            D = []
            Z = []
            if rho is not None and weight is not None:
                raise ValueError("Must use either rho or weights, not both")

            elif rho != None:
                s = KaplanMeier(Survival(t,censoring=censoring))
                s.fit()
                s = (s.results[0][0]) ** (rho)
                s = np.r_[1,s[:-1]]

            elif weight is not None:
                s = weight(tind)

            else:
                s = np.ones_like(tind)

            if censoring is None:
                ##Update with stratification
                for g in groups:
                    n = len(t)
                    if pooled.ndim == 1:
                        exog_idx = exog == g
                    else:
                        exog_idx = (np.product(exog == g, axis=1)).astype(bool)
                    dk = np.bincount(t[exog_idx])
                    d = np.bincount(t)
                    if np.max(tind) != len(dk):
                        dif = np.max(tind) - len(dk) + 1
                        dk = np.r_[dk,[0]*dif]
                    dk = dk[:,list(tind)]
                    d = d[:,list(tind)]
                    dk = dk.astype(float)
                    d = d.astype(float)
                    dkSum = np.cumsum(dk)
                    dSum = np.cumsum(d)
                    dkSum = np.r_[0,dkSum]
                    dSum = np.r_[0,dSum]
                    nk = len(exog[exog_idx]) - dkSum[:-1]
                    n -= dSum[:-1]
                    d = d[n>1]
                    dk = dk[n>1]
                    nk = nk[n>1]
                    n = n[n>1]
                    s = s[n>1]
                    ek = (nk * d)/(n)
                    Z.append(np.sum(s * (dk - ek)))
                    NK.append(nk)
                    N.append(n)
                    D.append(d)
            else:
                for g in groups:
                    if pooled.ndim == 1:
                        exog_idx = exog == g
                    else:
                        exog_idx = (np.product(exog == g, axis=1)).astype(bool)
                    reverseCensoring = -1*(censoring - 1) 
                    censored = np.bincount(t,reverseCensoring)
                    ck = np.bincount(t[exog_idx],
                                     reverseCensoring[exog_idx])
                    dk = np.bincount(t[exog_idx],
                                     censoring[exog_idx])
                    d = np.bincount(t,censoring)
                    if np.max(tind) != len(dk):
                        dif = np.max(tind) - len(dk) + 1
                        dk = np.r_[dk,[0]*dif]
                        ck = np.r_[ck,[0]*dif]
                    dk = dk[:,list(tind)]
                    ck = ck[:,list(tind)]
                    d = d[:,list(tind)]
                    dk = dk.astype(float)
                    d = d.astype(float)
                    ck = ck.astype(float)
                    dkSum = np.cumsum(dk)
                    dSum = np.cumsum(d)
                    ck = np.cumsum(ck)
                    ck = np.r_[0,ck]
                    dkSum = np.r_[0,dkSum]
                    dSum = np.r_[0,dSum]
                    censored = censored[:,list(tind)]
                    censored = censored.astype(float)
                    censoredSum = np.cumsum(censored)
                    censoredSum = np.r_[0,censoredSum]
                    nk = (len(exog[exog_idx]) - dkSum[:-1]
                          - ck[:-1])
                    n = len(censoring) - dSum[:-1] - censoredSum[:-1]
                    d = d[n>1]
                    dk = dk[n>1]
                    nk = nk[n>1]
                    n = n[n>1]
                    s = s[n>1]
                    ek = (nk * d)/(n)
                    Z.append(np.sum(s * (dk - ek)))
                    NK.append(nk)
                    N.append(n)
                    D.append(d)
                    self.nk = nk
                    self.d=d
                    self.n = n
                    self.dk = dk
                    self.ek = ek
                    self.testEx = exog
                    self.g = g
                    self.ein = exog_idx
            Z = np.array(Z)
            N = np.array(N)
            D = np.array(D)
            NK = np.array(NK)
            sigma = -1 * np.dot((NK/N) * ((N - D)/(N - 1)) * D
                                * np.array([(s ** 2)]*len(D))
                            ,np.transpose(NK/N))
            np.fill_diagonal(sigma, np.diagonal(np.dot((NK/N)
                                                  * ((N - D)/(N - 1)) * D
                                                       * np.array([(s ** 2)]*len(D))
                                                  ,np.transpose(1 - (NK/N)))))
            chisq = np.dot(np.transpose(Z),np.dot(la.pinv(sigma), Z))
            df = len(groups) - 1
            self.var = sigma
            self.N = N
            self.D = D
            self.NK = NK
            self.Z = Z
            return np.array([chisq, df, stats.chi2.sf(chisq,df)])
        else:
            raise ValueError("groups must be in column exog")

    def isolate_curve(self, exog):
        """
        Get results for one curve from a model that fits mulitple survival
        curves

        Parameters
        ----------

        exog: The value of that exogenous variable for the curve to be
        isolated.

        returns
        --------
        A SurvivalResults object for the isolated curve
        """

        exogs = self.exog
        if exog is None:
            raise ValueError("Already a single curve")
        else:
            ind = (list((self.model).groups)).index(exog)
            results = self.results[ind]
            ts = self.ts[ind]
            censoring = self.censoring[exogs == exog]
            censorings = self.censorings[ind]
            event = self.event[ind]
            r = KMResults(self.model, results[0], results[1])
            r.results = results
            ##Need to check
            r.ts = []
            r.ts.append(ts)
            r.censoring = censoring
            r.censorings = censorings
            r.event = event
            r.exog = None
            r.groups = None
            return r

    def plot(self, confidence_band=False):
        """
        Plot the estimated survival curves. After using this method
        do

        plt.show()

        to display the plot
        """
        plt.figure()
        if self.exog is None:
            self._plotting_proc(0, confidence_band)
        else:
            for g in range(len(self.groups)):
                self._plotting_proc(g, confidence_band)
        plt.ylim(ymax=1.05)
        plt.ylabel('Survival')
        plt.xlabel('Time')

    def summary(self):
        """
        Print a set of tables containing the estimates of the survival
        function, and its standard errors
        """
        if self.exog is None:
            self._summary_proc(0)
        else:
            for g in range(len(self.groups)):
                self._summary_proc(g)

    def _plotting_proc(self, g, confidence_band):
        """
        For internal use
        """
        survival = self.results[g][0]
        t = self.ts[g]
        e = (self.event)[g]
        if self.censoring is not None:
            c = self.censorings[g]
            csurvival = survival[c != 0]
            ct = t[c != 0]
            if len(ct) != 0:
                plt.vlines(ct,csurvival+0.02,csurvival-0.02)
        t = np.repeat(t[e != 0], 2)
        s = np.repeat(survival[e != 0], 2)
        if confidence_band:
            lower = self.results[g][2]
            upper = self.results[g][3]
            lower = np.repeat(lower[e != 0], 2)
            upper = np.repeat(upper[e != 0], 2)
        if self.ts[g][-1] in t:
            t = np.r_[0,t]
            s = np.r_[1,1,s[:-1]]
            if confidence_band:
                lower = np.r_[1,1,lower[:-1]]
                upper = np.r_[1,1,upper[:-1]]
        else:
            t = np.r_[0,t,self.ts[g][-1]]
            s = np.r_[1,1,s]
            if confidence_band:
                lower = np.r_[1,1,lower]
                upper = np.r_[1,1,upper]
        if confidence_band:
            plt.plot(t,(np.c_[lower,upper]),'k--')
        plt.plot(t,s)

    def _summary_proc(self, g):
        """
        For internal use
        """
        if self.exog is not None:
            myTitle = ('exog = ' + str(self.groups[g]) + '\n')
            table = np.transpose(self.results[g])
            table = np.c_[np.transpose(self.ts[g]),table]
            table = SimpleTable(table, headers=['Time','Survival','Std. Err',
                                                'Lower 95% CI', 'Upper 95% CI'],
                                title = myTitle)
        else:
            myTitle = "Kaplan-Meier Curve"
            table = np.transpose(self.results[0])
            table = np.c_[self.ts[0],table]
            table = SimpleTable(table, headers=['Time','Survival','Std. Err',
                                                'Lower 95% CI', 'Upper 95% CI'],
                                title = myTitle)
        print(table)

class CoxResults(LikelihoodModelResults):

    """
    Results for cox proportional hazard models
    """

    def test_coefficients(self, method="wald"):
        params = self.params
        model = self.model
        ##Other methods (e.g. score?)
        if method == "wald":
            se = 1/(np.sqrt(np.diagonal(model.information(params))))
            z = params/se
        return np.c_[params,se,z,2 * stats.norm.sf(np.abs(z), 0, 1)]

    def wald_test(self, restricted=None):
        if restricted is None:
            restricted = self.model.start_params
        params = self.params
        model = self.model
        return np.dot((np.dot(params - restricted, model.information(params)))
                      , params - restricted)

    def score_test(self, restricted=None):
        if restricted is None:
            restricted = self.model.start_params
        model = self.model
        score = model.score(restricted)
        cov = model.covariance(restricted)
        return np.dot(np.dot(score, cov), score)

    def likelihood_ratio_test(self, restricted=None):
        if restricted is None:
            restricted = self.model.start_params
        params = self.params
        model = self.model
        if isinstance(restricted, CoxResults):
            restricted = restricted.model.loglike(restricted.params)
            return 2 * (model.loglike(params) - restricted)
        else:
            return 2 * (model.loglike(params) - model.loglike(restricted))

    def conf_int(self, alpha=.05, cols=None, method='default', exp=True):
        CI = super(CoxResults, self).conf_int(alpha, cols, method)
        if exp:
            CI = np.exp(CI)
        return CI

