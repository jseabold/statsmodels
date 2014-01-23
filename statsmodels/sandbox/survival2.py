#Survival Analysis
from itertools import starmap
import numpy as np
import numpy.linalg as la
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from scipy import stats

from statsmodels.tools import data
from statsmodels.iolib.table import SimpleTable
from statsmodels.base.model import (Model, LikelihoodModel,
                                    LikelihoodModelResults)
from statsmodels.base.data import handle_data
from statsmodels.graphics import utils as graphics_utils
from statsmodels.tools.decorators import cache_readonly
from patsy import dmatrix
from pandas import Series, DataFrame, Index, isnull
try:
    from scipy.maxentropy import logsumexp as sp_logsumexp
except:
    from scipy.misc import logsumexp as sp_logsumexp

#TODO: turn this into a helper utility function - also used in graphics
def _maybe_name_or_idx_data(name, X):
    if name is None:
        return
    elif isinstance(name, basestring):
        return Series(dmatrix(name + "-1", X).squeeze(), name=name)
    elif isinstance(name, int):
        if data._is_using_ndarray(X, None):
            return X[:, name]
        elif data._is_using_pandas(X, None):
            return X[X.columns[name]]
        else:
            raise ValueError("Data of type %s not understood" % type(X))


def _maybe_asarray_pandas_passthru(X):
    if X is None:
        return
    elif data._is_using_pandas(X, None):
        if (X.ndim == 2 and X.shape[1] > 1):
            raise ValueError("DataFrame must contain one variable")
        else:
            return X
    else:
        return np.asarray(X)


def _maybe_event_to_censored(X):
    if X is None:
        return
    elif data._is_using_ndarray_type(X, None):
        return (1 - X)
    elif data._is_using_pandas(X, None):
        return Series((1 - X), name="censored")


### Generic base class


class Survival(object):
    """
    Create an object to store survival data for processing
    by other survival analysis functions

    Parameters
    -----------
    time1 : int or array-like
        if time2=None, index of column containing the
        duration that the subject survivals and remains
        uncensored (e.g. observed survival time), if
        time2 is not None, then time1 is the index of
        a column containing start times for the
        observation of each subject(e.g. observed survival
        time is end time minus start time)
    time2 : None, int or array-like
        index of column containing end times for each observation
    event : int or array-like
        index of the column containing an indicator of whether an observation
        is an event, or a censored observation, with 0 for censored, and 1
        for an event.
    data : array-like, optional
        An array, with observations in each row, and variables in the columns

    Attributes
    -----------
    times : array
        vector of survival times
    censored : array
        vector of censoring indicators. This is the opposite of the event.
    time_type : str
        indicator of what type of censoring occurs

    Examples
    ---------
    see other survival analysis functions for examples
    of usage with those functions
    """
    def __init__(self, time1, time2=None, event=None, ctype=None, data=None):
        if data is not None:
            event = _maybe_name_or_idx_data(event, data)
            censored = _maybe_event_to_censored(event)
            if time2 is not None:
                self.start = _maybe_name_or_idx_data(time1, data)
                self.end = _maybe_name_or_idx_data(time2, data)
                time = self.start - self.end
                self.time_type = "interval"
            else:
                time = _maybe_name_or_idx_data(time1, data)
                self.time_type = "exact"
        else:
            event = _maybe_asarray_pandas_passthru(event)
            censored = _maybe_event_to_censored(event)
            time1 = _maybe_asarray_pandas_passthru(time1)
            if time2 is not None:
                self.start = time1
                self.end = _maybe_asarray_pandas_passthru(time2)
                time = (self.start - self.end)
                self.time_type = "interval"
            else:
                self.time_type = "exact"
                time = time1
        names = ["time", "event", "censored"] # for correct expected order
        if event is None:
            event = [1] * len(time)
            censored = [0] * len(time)
        self.data = DataFrame.from_dict(dict(time=time,
                                             event=event,
                                             censored=censored))[names]
    def __len__(self):
        return len(self.data)

    @property
    def time(self):
        return self.data["time"]

    @property
    def event(self):
        return self.data["event"]

    @property
    def censored(self):
        return self.data["censored"]

    def mean(self):
        # hard-coded to act on axis 0
        return np.average(self.time, weights=self.event)


class SurvivalModel(Model):
    def __init__(self, surv, exog=None, groups=None, missing='none',
                       **kwargs):
        self.surv = surv

        # handle missing data in long form
        orig_data = handle_data(surv.data, exog, groups=groups,
                                missing=missing, **kwargs)

        # side effects attaches (collapsed) data, super call to Model, etc.
        self._init_survival_data(orig_data, **kwargs)

    def _init_survival_data(self, data, **kwargs):
        # attach after dropping missing data before collapsing
        self.surv = Survival(data.endog[:,0],
                             event=data.endog[:,1])
        ynames = data.ynames
        group_names = data.group_names
        all_data = data.data_with_groups
        #TODO: you don't need all this outside of Kaplan Meier
        if data.groups is not None:
            group_nobs = all_data.groupby(group_names).size()
            # group by time and groups
            grps = all_data.groupby([ynames[0]] + group_names)
            #NOTE: what is this going to do when there are multiple group
            # variables?
            collapsed = grps.sum()[ynames[1:]].sortlevel(level=1)
            # cumulative events and censored by each group over time
            cumsum_shift = lambda x : x.cumsum().shift()
            nrisk = -collapsed.groupby(level=1).apply(cumsum_shift).fillna(0)
            # shift the observations up a period to get number at risk
            #TODO: might have to change this assumption for left- vs. right-
            #continuous intervals
            nrisk = nrisk.sum(1)
            collapsed["nrisk"] = group_nobs.add(nrisk, level=1)
        else:
            group_nobs = len(all_data)
            collapsed = all_data.groupby(ynames[0]).sum()
            cumsum_shift = lambda x : x.cumsum().shift()
            nrisk = collapsed.apply(cumsum_shift).fillna(0).sum(1)
            collapsed["nrisk"] = group_nobs - nrisk

        #NOTE: older version of pandas are going to return something here...
        collapsed.reset_index(inplace=True)
        self.collapsed_data = collapsed[ynames].values
        #self.nrisk = collapsecensoredd["nrisk"]
        if group_names:
            self.group_names = group_names
            self.collapsed_groups = collapsed[group_names]
        else:
            self.group_names = None
            self.collapsed_groups = None

        #NOTE: event and censored need an asarray - can attach original
        #to SurvivalData
        # need to repass the data, because Grouping magics are at the
        # ModelData level, missing has already been handled
        # by doing this, we're never going to get a PandasGroupedData object
        #import ipdb; ipdb.set_trace()
        #super(SurvivalModel, self).__init__(data.endog.astype(float),
        #                                    data.exog,
        #                                    groups=data.groups,
        #                                    missing='none', **kwargs)

        #import ipdb; ipdb.set_trace()
        #TODO: don't attach endog like this
        self.data = data
        self.endog = data.endog
        self.exog = data.exog
        self.time = self.endog[:,0]
        self.event = self.endog[:,1]
        self.censored = self.endog[:,2]
        self.group_names = self.data.group_names

#TODO: make sure data is converted to floats, we don't use bincount anymore

# it may be the case that nrisk, events, etc. just needs an index of
# time and group
class KaplanMeier(SurvivalModel):
    """
    Create an object of class KaplanMeier for estimating
    Kaplan-Meier survival curves.

    TODO: parts of docstring are outdated

    Parameters
    ----------
    data : array-like
        An array, with observations in each row, and
        variables in the columns
    surv : Survival object
        Survival object containing desire times and censoring
    endog : int or array-like
        index (starting at zero) of the column
        containing the endogenous variable (time),
        or if endog is an array, an array of times
        (in this case, data should be none)
    exog : int or array-like
        index of the column containing the exogenous
        variable (must be categorical). If exog = None, this
        is equivalent to a single survival curve. Alternatively,
        this can be a vector of exogenous variables index in the same
        manner as data provided either from data or surv
        or if exog is an array, an array of exogenous variables
        (in this case, data should be none)
    censoring : int or array-like
        index of the column containing an indicator
        of whether an observation is an event, or a censored
        observation, with 0 for censored, and 1 for an event
        or if censoring is an array, an array of censoring
        indicators (in this case, data should be none)

    Attributes
    -----------
    censorings : array
        List of censorings associated with each unique
        time, at each value of exog
    events : array
        List of the number of events at each unique time
        for each value of exog
    results : array
        List of arrays containing estimates of the value
        value of the survival function and its standard error
        at each unique time, for each value of exog
    ts : array
        List of unique times for each value of exog

    Methods
    -------
    fit : Calcuate the Kaplan-Meier estimates of the survival
        function and its standard error at each time, for each
        value of exog

    Examples
    --------
    TODO: interface, argument list is outdated
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
    >>> results = km.fit()
    >>> results.plot()

    results is a KMResults object

    Doing

    >>> results.summary()

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
    >>> results2 = km2.fit()

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
    >>> results3 = km3.fit()

    Test for difference of survival curves

    >>> log_rank = results3.test_diff([0.0645,-0.03957])

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
    >>> results4 = km4.fit()

    """

    ##Rework interface and data structures?
    ##survival attribute?

    ##Add stratification

    ##update usage with Survival for changes to Survival

    def __init__(self, surv, groups=None, missing='none'):
        self.time_type = surv.time_type
        # this has side effects, attaches collapsed data and nrisk
        super(KaplanMeier, self).__init__(surv, None, groups=groups,
                                          missing=missing, hasconst=False)

    def _init_survival_data(self, data, **kwargs):
        ynames = data.ynames
        group_names = data.group_names
        all_data = data.data_with_groups

        # calculate nrisk by group and collapse the data because exog is None
        if data.groups is not None:
            group_nobs = all_data.groupby(group_names).size()
            # group by time and groups
            grps = all_data.groupby([ynames[0]] + group_names)
            #NOTE: what is this going to do when there are multiple group
            # variables?
            collapsed = grps.sum()[ynames[1:]].sortlevel(level=1)
            # cumulative events and censored by each group over time
            cumsum_shift = lambda x : x.cumsum().shift()
            nrisk = -collapsed.groupby(level=1).apply(cumsum_shift).fillna(0)
            # shift the observations up a period to get number at risk
            #TODO: might have to change this assumption for left- vs. right-
            #continuous intervals
            nrisk = nrisk.sum(1)
            collapsed["nrisk"] = group_nobs.add(nrisk, level=1)
        else:
            group_nobs = len(all_data)
            collapsed = all_data.groupby(ynames[0]).sum()
            cumsum_shift = lambda x : x.cumsum().shift()
            nrisk = collapsed.apply(cumsum_shift).fillna(0).sum(1)
            collapsed["nrisk"] = group_nobs - nrisk

        collapsed = collapsed.reset_index(inplace=True)
        self.collapsed_data = collapsed[ynames]
        self.nrisk = collapsed["nrisk"]
        self.event = collapsed["event"]
        self.time = collapsed["time"]
        if group_names:
            self.groups = collapsed[group_names]
            self.group_names = group_names
        else:
            self.group_names = None
            self.groups = None

        # initialize the data again because grouping magic is at data level
        super(SurvivalModel, self).__init__(collapsed[ynames + ["nrisk"]],
                                            None,
                                            groups=self.groups,
                                            missing='none', **kwargs)

    #def from_formula(self, formula, data, subset):
    #    pass

    def fit(self):
        """
        Calculate the Kaplan-Meier estimator of the survival function

        Returns
        -------
        KMResults instance for the estimated survival curve(s)
        """
        event = self.event
        nrisk = self.nrisk
        hazard = event/nrisk
        if self.groups is not None:
            #TODO: make sure this stuff is arrays internally already
            #TODO: I should just be able to call group(array) and have the
            # grouped results. This should be part of the Model API
            data_dict = dict(survival = 1 - np.asarray(hazard),
                             nrisk=self.nrisk, event=self.event)
            data_dict.update(zip(self.group_names, np.asarray(self.groups).T))
            data = DataFrame.from_dict(data_dict)
            grouped = data.groupby(self.group_names)
            survival = grouped["survival"].cumprod().values
            #from IPython.core.debugger import Pdb; Pdb().set_trace()
            def get_std_err(x):
                return (x["survival"].cumprod()**2 *
                        (x["event"]/
                         (x["nrisk"]*(x["nrisk"]-x["event"]))).cumsum()
                        )**.5
            std_err = grouped.apply(get_std_err)
        else:
            survival = np.cumprod(1 - hazard)
            std_err = (survival**2 *
                       np.cumsum(event/(nrisk*(nrisk-event))))**.5

        # variance calculated according to Greenwood (1926)
        results = np.c_[hazard, survival, std_err]
        return KMResults(self, results)


def get_td(data, ntd, td, td_times, censoring=None, times=None,
           ntd_names=None, td_name=None):
    """
    For fitting a Cox model with a time-dependent covariate.
    Split the data into intervals over which the covariate
    is constant

    Parameters
    ----------
    data : array
        array containing the all variables to be used
    ntd : list
        list of indices in data of the non-time-dependent
        covariates
    td : list
        list of indices of the time-dependent covariate in data.
        Each column identified in data is interpreted as the value
        of the covariate at a secific time (specified by td_times)
    td_times : array
        array of times associated with each column identified by td
    censoring : int
        index of the censoring indicator in data
    times : int
        only need if censoring is not none. Index of times for
        the original observations that occur in data
    ntd_names : array
        array of names for the non-time-dependent variables.
        This is useful, since the ordering of the variables
        is not preserved
    td_name : array (containing only one element)
        array containing the name of the newly created time-dependent
        variable

    Returns
    -------
    If no names are given, a 2d array containing the data in
    time-dependent format. If names are given, the first return is
    the same as previous, and the second return is an array of names

    """
    ##Add names
    ##Check results
    ##Add lag
    ##Do without data?
    ##For arbitrarily many td vars


    ntd = data[:,ntd]
    td = data[:,td]
    ind = ~np.isnan(td)
    rep = ind.sum(1)
    td_times = np.repeat(td_times[:,np.newaxis], len(td), axis=1)
    td = td.flatten()
    ind = ~np.isnan(td)
    td = td[ind]
    td_times = td_times.flatten('F')[ind]
    start = np.r_[0,td_times[:-1]]
    ##Does the >= solve the underlying problem?
    start[start >= td_times] = 0
    ntd = np.repeat(ntd, rep, axis=0)
    if censoring is not None:
        censoring = data[:,censoring]
        times = data[:,times]
        censoring = np.repeat(censoring, rep)
        times = np.repeat(times, rep)
        ind = ((td_times == times) * censoring) != 0
        censoring[ind] = 1
        censoring[~ind] = 0
        if ntd_names is not None:
            return (np.c_[start,td_times,censoring,ntd,td],
                    np.r_[np.array(['start','end','censoring'])
                          ,ntd_names,td_name])
        else:
            return np.c_[start,td_times,censoring,ntd,td]
    else:
        if ntd_names is not None:
            return (np.c_[start, td_times, ntd, td],
                    np.r_[np.array(['start','end']),ntd_names,td_name])
        else:
            return np.c_[start, td_times, ntd, td]


def _loglike_breslow_exact(params, exog, times, event_idx, collapsed_data):
    """
    exog, times, and collapsed_data should be for each strata
    """
    fittedvalues = np.dot(exog, params)
    exp_fittedvalues = np.exp(fittedvalues)
    # drop observations with no events
    collapsed_data = collapsed_data[collapsed_data[:,1] != 0]

    # default partial likelihood
    logL = np.dot(fittedvalues, event_idx)
    for time_i, tied in collapsed_data:
        at_risk = times >= time_i
        logL -= sp_logsumexp(fittedvalues[at_risk]) * tied
    return logL


def _loglike_efron_exact(params, exog, times, event_idx, collapsed_data):
    fittedvalues = np.dot(exog, params)
    exp_fittedvalues = np.exp(fittedvalues)
    # drop observations with no events
    collapsed_data = collapsed_data[collapsed_data[:,1] != 0]

    # default partial likelihood
    logL = np.dot(fittedvalues, event_idx)
    for time_i, tied in collapsed_data:
        #event at time_i
        ind = np.logical_and(event_idx, times == time_i)
        at_risk = times >= time_i
        logL -= np.sum(np.log(np.dot(exp_fittedvalues, at_risk)
                - np.arange(tied)/tied * np.dot(exp_fittedvalues, ind)))
    return logL


_coxph_loglike_funcs = {
                  "efron" : _loglike_efron_exact,
                  "breslow" : _loglike_breslow_exact,
        }


def _score_breslow(params, exog, times, event_idx, collapsed_data):
    score = np.sum(exog[event_idx], axis=0)
    fittedvalues = np.dot(exog, params)
    exp_fittedvalues = np.exp(fittedvalues)
    for time_i, tied in collapsed_data:
        at_risk = times >= time_i
        exp_fittedvalues_j = exp_fittedvalues[at_risk]
        X_j = exog[at_risk]
        score -= tied * (np.dot(exp_fittedvalues_j, X_j)/
                         np.sum(exp_fittedvalues_j))

    return score


def _score_efron(params, exog, times, event_idx, collapsed_data):
    fittedvalues = np.dot(exog, params)
    exp_fittedvalues = np.exp(fittedvalues)
    # drop those that didn't have an event
    collapsed_data = collapsed_data[collapsed_data[:,1] != 0]
    score = np.sum(exog[event_idx], axis=0)
    for time_i, tied in collapsed_data:
        had_event_i = np.logical_and(event_idx, times == time_i)
        at_risk = times >= time_i
        exp_fittedvalues_j = exp_fittedvalues[at_risk]
        exp_fittedvalues_i = exp_fittedvalues[had_event_i]
        num1 = np.dot(exp_fittedvalues_j, exog[at_risk])
        num2 = np.dot(exp_fittedvalues_i, exog[had_event_i])
        de1 = exp_fittedvalues_j.sum()
        de2 = exp_fittedvalues_i.sum()
        c = (np.arange(tied)/tied)[:,None]
        score -= np.sum((num1 - c * num2) / (de1 - c * de2), axis=0)
    return score


_coxph_score_funcs = {"efron" : _score_efron,
                      "breslow" : _score_breslow}


def _hessian_efron(params, exog, times, event_idx, collapsed_data):
    params = np.atleast_1d(params) # powell seems to squeeze, messing up dims
    fittedvalues = np.dot(exog, params)
    exp_fittedvalues = np.exp(fittedvalues)
    collapsed_data = collapsed_data[collapsed_data[:,1] != 0]
    hess = 0

    for time_i, tied in collapsed_data:
        had_event_i = np.logical_and(event_idx, times == time_i)
        at_risk = times >= time_i
        exp_fittedvalues_j = exp_fittedvalues[at_risk]
        Xj = exog[at_risk]
        exp_fittedvalues_i = exp_fittedvalues[had_event_i]
        Xi = exog[had_event_i]
        exp_fittedvalues_Xj = np.dot(exp_fittedvalues_j, Xj)
        exp_fittedvalues_Xi = np.dot(exp_fittedvalues_i, Xi)
        num1 = np.dot(Xj.T, (Xj * exp_fittedvalues_j[:, None]))
        num2 = np.dot(Xi.T, (Xi * exp_fittedvalues_i[:, None]))
        de1 = exp_fittedvalues_j.sum()
        de2 = exp_fittedvalues_i.sum()

        #TODO: replace this with broadcasting
        for i in range(int(tied)):
            c = i/float(tied)
            num3 = (exp_fittedvalues_Xj - c * exp_fittedvalues_Xi)
            de = de1 - c * de2
            hess += (((num1 - c * num2) / (de)) -
                     (np.dot(num3[:, None], num3[None,:])
                      / (de**2)))
    return hess
#return np.atleast_2d(hess)


def _hessian_breslow(params, exog, times, event_idx, collapsed_data):
    params = np.atleast_1d(params) # powell seems to squeeze, messing up dims
    fittedvalues = np.dot(exog, params)
    exp_fittedvalues = np.exp(fittedvalues)
    collapsed_data = collapsed_data[collapsed_data[:,1] != 0]
    hess = 0
    for time_i, tied in collapsed_data:
        at_risk = times >= time_i
        exp_fittedvalues_j = exp_fittedvalues[at_risk]
        Xj = exog[at_risk]
        exp_fittedvalues_X = np.mat(np.dot(exp_fittedvalues_j, Xj))
        #TODO: Save more variables to avoid recalulation?
        hess += ((((np.dot(Xj.T, (Xj * exp_fittedvalues_j[:, None])))/
                    (exp_fittedvalues_j.sum()))
                 - ((np.array(exp_fittedvalues_X.T * exp_fittedvalues_X))/((exp_fittedvalues_j.sum())**2))) * tied)
    return hess

_coxph_hessian_funcs = {"efron" : _hessian_efron,
                        "breslow" : _hessian_breslow}


def _coxph_group_arguments_factory(model, params):
    """
    Returns a generator that yields the arguments that are necessary
    for all the loglike, score, and hessian functions for CoxPH.
    """
    for group_name, endog, exog in model.data.group_by_groups():
        times = endog[:,0]
        event_idx = endog[:,1].astype(bool)
        collapsed_groups = model.collapsed_groups
        if group_name is not None:
            #TODO: make sure this an array anyway
            collapsed_idx = np.array(collapsed_groups == group_name).squeeze()
            collapsed_data = model.collapsed_data[collapsed_idx]
        else:
            collapsed_data = model.collapsed_data
        yield params, exog, times, event_idx, collapsed_data[:,:2]


class CoxPH(SurvivalModel, LikelihoodModel):
    """
    Fit a cox proportional harzard model from survival data

    Parameters
    ----------
    surv : Survival object
        Survival object with the desired times and censoring
    exog : int or array-like
        if data is not None, index or list of indicies of data
        for the columns of the desired exogenous variables
        if data is None, then a 2d array of the desired
        exogenous variables
    data : array-like
        optional array from which the exogenous variables will
        be selected from the indicies given as exog
    ties : string
        A string indicating the method used to handle ties
    strata : array-like
        optional, if a stratified cox model is desired.
        list of indicies of columns of the matrix of exogenous
        variables that are to be included as strata. All other
        columns will be included as unstratified variables
        (see documentation for statify method)

    Attributes:
    -----------
    surv : The initial survival object given to CoxPH
    ties : String indicating how to handle ties
    censoring : Vector of censoring indicators
    ttype : String indicating the type of censoring
    exog : The 2d array of exogenous variables
    strata : Indicator of how, if at all, the model is stratified
    d :  For exact times, a 2d array, whose first column is the
        unique times, and whose second column is the number of ties
        at that time. For interval times, a 2d array where each
        row is one of the unique intervals

    Examples
    --------

    References
    ----------

    D. R. Cox. "Regression Models and Life-Tables",
        Journal of the Royal Statistical Society. Series B (Methodological)
        Vol. 34, No. 2 (1972), pp. 187-220
    """

    ##Handling for time-dependent covariates
    ##Handling for time-dependent coefficients
    ##Interactions
    ##Add residuals

    def __init__(self, surv, exog, groups=None, ties="efron", missing='none'):
        super(CoxPH, self).__init__(surv, exog, groups=groups, missing=missing,
                                    hasconst=None)
        #TODO: what to do with hasconst?
        self.ties = ties
        self.confint_dist = stats.norm
        self.exog_mean = self.exog.mean(axis=0)

    def information(self, params):
        """
        Calculate the Fisher information matrix at estimates of the
        parameters

        Parameters
        ----------
        params : estimates of the model parameters

        Returns
        -------
        information matrix as 2d array
        """
        return -self.hessian(params)

    def loglike(self, params):
        # use built-in sum on generator
        return sum(starmap(_coxph_loglike_funcs[self.ties],
                              _coxph_group_arguments_factory(self, params)))

    def score(self, params):
        # use built-in sum on generator
        return sum(starmap(_coxph_score_funcs[self.ties],
                              _coxph_group_arguments_factory(self, params)))

    def hessian(self, params):
        # use built-in sum on generator
        return -sum(starmap(_coxph_hessian_funcs[self.ties],
                              _coxph_group_arguments_factory(self, params)))

    def fit(self, start_params=None, method='newton', maxiter=100,
            full_output=1, disp=1, callback=None, retall=0, **kwargs):
        if start_params is None:
            self.start_params = np.zeros_like(self.exog[0])
        else:
            self.start_params = start_params

        fargs = (self.exog, self.time, self.event.astype(bool),
                self.collapsed_data[:,:2])

        #TODO: this magic is going to screw up the docs.
        ties = self.ties
        #self.loglike = lambda x : _coxph_loglike_funcs[ties](self, x, *fargs)
        #self.score = lambda x : _coxph_score_funcs[ties](self, x, *fargs)
        #self.hessian = lambda x : -_coxph_hessian_funcs[ties](self, x, *fargs)

        results = super(CoxPH, self).fit(start_params, method=method,
                        maxiter=maxiter, full_output=full_output,
                        disp=disp, callback=callback, retall=retall, **kwargs)
        return CoxResults(self, results.params)


def unstack_groups(X, group_keys, group_names, ynames, join_char="_"):
    """
    Takes grouped data and unstacked creating new variables ie.,
    time, event, group
    1,    1,     1
    2,    0,     1
    2,    1,     2

    becomes

    time, event_1, event_2
    1,    1,       nan
    2,    0,       1

    In this example, "time" is the non-unique index, group_keys is [1,2],
    group_names is ["group"], and ynames is ["event"]
    """
    for name in group_keys:
        for yname in ynames:
            new_name = yname + join_char + join_char.join(str(name))
            X[new_name] = np.nan
            X.ix[X[group_names] == name, new_name] = X[yname]
    return X


class KMResults(object):
    """
    Results for a Kaplan-Meier model

    Methods
    -------
    plot: Plot the survival curves using matplotlib.plyplot
    summary: Display the results of fit in a table. Gives results
        for all (including censored) times

    test_diff: Test for difference between survival curves

    TODO: drop methods from docstring,
    TODO: what is results attribute? document attributes

    """

    ##Add handling for stratification

    def __init__(self, model, results):
        self.model = model
        self.hazard = results[:,0]
        self.survival = results[:,1]
        self.std_err = results[:, 2] # std err of survival function
        self.groups = model.data.groups

    @cache_readonly
    def cumhazard(self):
        """
        Cumulative hazard function as defined by Peterson as -log(survival)
        """
        return -np.log(self.survival)

    @cache_readonly
    def std_err_cumhazard(self):
        """
        The standard error of the cumulative hazard function.
        """
        #NOTE: the cumulative hazard function is assumed to be
        # -log(survival) where survival is calculated using the product-limit
        # estimate of Peterson rather than the Nelson-Aalen estimate
        return ((self.std_err ** 2)/self.survival**2)**.5

    def conf_int_hazard(self, alpha=.05):
        """
        Confidence intervals for the cumulative hazard function

        Parameters
        ----------
        alpha : float
            Confidence interval level
        transform : string, "log" or "cloglog"
            The type of transformation used to keep the
            confidence interval in the interval [0,1].
            "log" applies the natural logarithm,
            "cloglog" applies the complementary cloglog log(-log(x))
        force : bool
            indicator of whether confidence interval values
            that fall outside of [0,1] should be forced to
            one of the endpoints
        """
        survival = self.survival
        std_err = self.std_err_hazard

        cut_off = stats.norm.ppf(1-alpha/2.)
        if transform == "log":
            lower = np.exp(np.log(survival) - cut_off * std_err *
                                                1/survival)
            upper = np.exp(np.log(survival) + cut_off * std_err *
                                                1/survival)
        elif transform == "cloglog":
            lower = np.exp(-np.exp(np.log(-np.log(survival)) - cut_off *
                                std_err * 1/(survival * np.log(survival))))
            upper = np.exp(-np.exp(np.log(-np.log(survival)) + cut_off *
                                std_err * 1/(survival * np.log(survival))))
        if force:
            lower[lower < 0] = 0
            upper[upper > 1] = 1

        return np.c_[lower, upper]

    #TODO: use patsy here?
    #def test_diff(self, groups, rho=None, weight=None):
    #    """
    #    Test for difference between survival curves

    #    Parameters
    #    ----------
    #    groups : list
    #        A list of the values for exog to test for difference.
    #        tests the null hypothesis that the survival curves for all
    #        values of exog in groups are equal
    #    rho : int in [0,1]
    #        compute the test statistic with weight S(t)^rho, where
    #        S(t) is the pooled estimate for the Kaplan-Meier survival
    #        function.
    #        If rho = 0, this is the logrank test, if rho = 0, this is the
    #        Peto and Peto modification to the Gehan-Wilcoxon test.
    #    weight : function
    #        User specified function that accepts as its sole arguement
    #        an array of times, and returns an array of weights for each time
    #        to be used in the test

    #    Returns
    #    -------
    #    res : ndarray
    #        An array whose zeroth element is the chi-square test statistic for
    #        the global null hypothesis, that all survival curves are equal,
    #        the index one element is degrees of freedom for the test, and the
    #        index two element is the p-value for the test.

    #    Examples
    #    --------

    #    >>> import statsmodels.api as sm
    #    >>> import matplotlib.pyplot as plt
    #    >>> import numpy as np
    #    >>> from statsmodels.sandbox.survival2 import KaplanMeier
    #    >>> dta = sm.datasets.strikes.load()
    #    >>> dta = dta.values()[-1]
    #    >>> censoring = np.ones_like(dta[:,0])
    #    >>> censoring[dta[:,0] > 80] = 0
    #    >>> dta = np.c_[dta,censoring]
    #    >>> km = KaplanMeier(dta, 0, exog=1, censoring=2)
    #    >>> results = km.fit()

    #    Test for difference of survival curves

    #    >>> log_rank = results.test_diff([0.0645,-0.03957])

    #    The zeroth element of log_rank is the chi-square test statistic
    #    for the difference between the survival curves using the log rank test
    #    for exog = 0.0645 and exog = -0.03957, the index one element
    #    is the degrees of freedom for the test, and the index two element
    #    is the p-value for the test

    #    >>> wilcoxon = results.test_diff([0.0645,-0.03957], rho=1)

    #    wilcoxon is the equivalent information as log_rank, but for the
    #    Peto and Peto modification to the Gehan-Wilcoxon test.

    #    User specified weight functions

    #    >>> log_rank = results.test_diff([0.0645,-0.03957], weight=np.ones_like)

    #    This is equivalent to the log rank test

    #    More than two groups

    #    >>> log_rank = results.test_diff([0.0645,-0.03957,0.01138])

    #    The test can be performed with arbitrarily many groups, so long as
    #    they are all in the column exog

    #    """

    def test_diff(self, groups=None, test="logrank"):
        model = self.model
        if groups == None:
            group_keys = model.data.group_keys
        else:
            #TODO: do some error checking here
            group_keys = groups
        n_groups = len(group_keys)

        group_names = model.group_names
        if len(group_names) == 1:
            group_names = group_names[0]

        ynames = model.endog_names
        unstacked = model.data.data_with_groups.groupby(ynames[0]).apply(
                        unstack_groups, group_keys, group_names, ynames[1:])

        new_names = unstacked.columns.diff(ynames[1:])
        unstacked = unstacked[new_names]

        result = None
        for name, group in unstacked.groupby(group_names):
            group.set_index(ynames[0], inplace=True)
            if result is not None:
                result = result.reindex(result.index.union(group.index))
                result.update(group)
            else:
                result = group

        # if any events are nans they're changed to zeros
        event = result.filter(regex=ynames[1]+".*").fillna(0)
        nrisk = result.filter(regex="nrisk.*")
        # back fill the nrisk columns
        nrisk.fillna(method="backfill", inplace=True)
        null_idx = isnull(nrisk)
        if np.any(null_idx):
            #TODO: there has GOT to be a better way to do this
            # we need to replace the first NaN at the end of the time series
            # with the last value minus the event number at the last value
            # then forward fill
            col_idx = np.where(null_idx.any(0))[0].tolist()
            row_idx = [np.where(i)[0][0] for _,i in null_idx.T.iterrows() if
                       np.where(i)[0].size]
            # don't go negative
            new_vals = np.maximum(0, (nrisk.shift(1).values -
                        event.shift(1).values)[row_idx, col_idx])
            nrisk.values[row_idx, col_idx] = new_vals
            nrisk.fillna(method="pad", inplace=True)

        # change them all back to arrays
        nrisk = nrisk.values # all the at risks for each group
        nrisk_j = nrisk.sum(1)[:,None] # all the at risks across groups
        observed = event.sum().values
        event = event.values # all the events for each group
        event_j = event.sum(1)[:,None] # all the events across groups

        if test.lower() == "wilcoxon":
            weights = nrisk_j
        elif test.lower() == "logrank":
            weights = 1.
        elif test.lower() == "tware":
            weights = nrisk_j ** .5
        else:
            if test.lower() == "peto":
                # modified survival function
                mod_hazard = event_j / (nrisk_j + 1)
                weights = np.cumprod(1 - mod_hazard)[:,None]
            elif test.lower().startswith("fh"):
                survival = KaplanMeier(self.model.surv).fit().survival[:,None]
                import re
                try:
                    p, q = re.match("fh\((\d+),\s*(\d+)\)", test).groups()
                except:
                    raise ValueError("Syntax not understood %s" % test)
                p, q = map(float, [p, q])
                weights = np.r_[[[1]],
                                survival[:-1] ** p * (1 - survival[:-1])**q]
            else:
                raise ValueError("Test %s not understood" % test)

        expected = nrisk / nrisk_j * event_j
        obs_exp = np.sum(weights*(event - expected), 0)
        O_E2_div_E = obs_exp**2/expected.sum(0)

        #sign is odd but this passes, nan_to_num because can get div by zero
        V = -np.dot((weights**2*nrisk).T,
                    np.nan_to_num(nrisk * event_j * (nrisk_j - event_j)/
                                  (nrisk_j**2 * (nrisk_j - 1))))
        V_diag = np.nansum(weights**2*nrisk * (nrisk_j - nrisk) * event_j *
                           (nrisk_j - event_j)/(nrisk_j**2 * (nrisk_j - 1)),
                           0)
        np.fill_diagonal(V, V_diag)

        chi2_stat = np.dot(np.dot(obs_exp, np.linalg.pinv(V)), obs_exp)
        dof = n_groups - 1
        pvalue = stats.chi2.sf(chi2_stat, dof)
        dataframe = DataFrame(np.empty((n_groups, 5)),
                                     columns=["nobs", "observed", "expected",
                                              "(O-E)**2/E", "(O-E)**2/V"],
                                     index=Index(group_keys,
                                                        name=group_names))
        dataframe.ix[:, "nobs"] = nrisk[0]
        dataframe.ix[:, "observed"] = observed
        dataframe.ix[:, "expected"] = expected.sum(0)
        dataframe.ix[:, "(O-E)**2/E"] = O_E2_div_E
        dataframe.ix[:, "(O-E)**2/V"] = obs_exp**2/V_diag
        test_res = {"chi2" : chi2_stat, "dof" : dof, "pvalue" : pvalue}
        return test_res, dataframe

    def get_curve(self, group=None):
        """
        Get results for one curve from a model that fits multiple survival
        curves

        Parameters
        ----------
        exog : float or int
            The value of that exogenous variable for the curve to be
            isolated.

        Returns
        -------
        kmres : KMResults instance
            A KMResults instance for the isolated curve
        """
        if self.model.exog is None:
            raise ValueError("Already a single curve")

        endog, exog = self.model.data.get_group(group)
        time, event, censored = endog.T
        return KMResults(KaplanMeier(Survival(time, event=event),
                                     exog=[group]*len(time),
                #TODO: do we attach missing attribute anywhere? we need to.
                                     missing='none'),
                                     {group : self.results[group]})

    def conf_int(self, alpha=.05, transform="log", force=True):
        """
        Parameters
        ----------
        alpha : float
            Confidence interval level
        transform : string, "log" or "cloglog"
            The type of transformation used to keep the
            confidence interval in the interval [0,1].
            "log" applies the natural logarithm,
            "cloglog" applies the complementary cloglog log(-log(x))
        force : bool
            indicator of whether confidence interval values
            that fall outside of [0,1] should be forced to
            one of the endpoints
        """
        survival = self.survival
        std_err = self.std_err

        cut_off = stats.norm.ppf(1-alpha/2.)
        if transform == "log":
            lower = np.exp(np.log(survival) - cut_off * std_err *
                                                1/survival)
            upper = np.exp(np.log(survival) + cut_off * std_err *
                                                1/survival)
        elif transform == "cloglog":
            lower = np.exp(-np.exp(np.log(-np.log(survival)) - cut_off *
                                std_err * 1/(survival * np.log(survival))))
            upper = np.exp(-np.exp(np.log(-np.log(survival)) + cut_off *
                                std_err * 1/(survival * np.log(survival))))
        if force:
            # alternatively, we could use Kalbfleisch and Prentice (1980).
            # to get MLE
            lower[lower < 0] = 0
            upper[upper > 1] = 1

        return np.c_[lower, upper]

    def plot(self, confidence_band=None, ci_transform="log", alpha=.05,
                   ax=None, **plot_kwargs):
        """
        Plot the estimated survival curves.

        Parameters
        ----------
        confidence_band : bool
            Whether confidence bands should be plotted. If None, the default
            is used. If there are no groups, confidence bands are plotted.
            If the data is grouped, then the default is not to plot
            confidence bandas.
        ci_transform : {"log", "cloglog"}

        alpha : float

        ax : matplotlib.Axes, optional
            Existing Axes instance.
        plot_kwargs


        Returns
        -----
        figure : matplotlib.Figure
            Figure instance.
        """
        fig, ax = graphics_utils.create_mpl_ax(ax)

        if confidence_band or (confidence_band == None and
                               self.model.data.groups is None):
            conf_int = self.conf_int(alpha=alpha, transform=ci_transform)
        else:
            conf_int = False
            confidence_band = None

        for group, endog, _ in self.model.data.group_by_groups():
            time, event, censored = endog.T
            if group is None:
                survival = self.survival
                if conf_int is not False:
                    confidence_band = conf_int
            else:
                idx = np.squeeze(self.groups == group)
                survival = self.survival[idx]
                if conf_int is not False:
                    confidence_band = conf_int[idx]
            self._plot_curve(survival, time, event, censored,
                             confidence_band, ax, linestyle='steps-post-',
                             **plot_kwargs)


        ax.set_ylim(-.05, 1.05)
        ax.set_xlim(0, self.model.time.max() * 1.05)
        ax.margins(.05, 0)
        ax.set_ylabel('Survival')
        ax.set_xlabel('Time')
        return fig

    def _plot_curve(self, survival, time, event, censored, confidence_band,
                    ax, **plot_kwargs):
        """
        plot the survival curve for a given group

        Parameters
        ----------
        g : int
            index of the group whose curve is to be plotted

        confidence_band : bool
            If true, then the confidence bands will also be plotted.

        """

        if censored is not None:
            # remove the values that are censored but an event occurs
            cidx = (censored - np.logical_and(event, censored)).astype(bool)
            csurvival = survival[cidx]
            ctime = time[cidx]
            ax.plot(ctime, csurvival, '+', markersize=15)

        ax.plot(np.r_[0, time], np.r_[1, survival], color='k', **plot_kwargs)

        if confidence_band is not None:
            lower, upper = confidence_band.T
            ax.plot(np.r_[0, time], np.r_[1, lower],
                    linestyle="steps-post--", color='b')
            ax.plot(np.r_[0, time], np.r_[1, upper],
                    linestyle="steps-post--", color='b')

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

    def _summary_proc(self, g):
        """
        display the summary of the survival curve for the given group

        Parameters
        ----------
        g : int
            index of the group to be summarized

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
        return table


class CoxResults(LikelihoodModelResults):
    """
    Results for cox proportional hazard models

    Attributes
    ----------
    model : CoxPH instance
        the model that was fit
    params : array
        estimate of the parameters
    normalized_cov_params : array
        variance-covariance matrix evaluated at params
    scale : float
        see LikelihoodModelResults
    exog_mean : array
        mean vector of the exogenous variables
    names : array
        array of names for the exogenous variables
    """

    def __init__(self, model, params, normalized_cov_params=None, scale=1.0):
        super(CoxResults, self).__init__(model, params, normalized_cov_params,
                                        scale)
        self.exog_mean = model.exog_mean

    def cov_params(self, params):

        """
        Calculate the covariance matrix at estimates of the
        parameters

        Parameters
        ----------

        params : estimates of the model parameters

        Returns
        -------

        covariance matrix as 2d array

        """
        return np.linalg.pinv(self.model.information(params))

    def summary(self):

        """
        Print a set of tables that summarize the Cox model

        """

        params = self.params
        exog_names = self.model.exog_names
        coeffs = np.c_[exog_names, self.test_coefficients()]
        coeffs = SimpleTable(coeffs, headers=['variable','parameter',
                                              'standard error', 'z-score',
                                              'p-value'],
                             title='Coefficients')
        CI = np.c_[exog_names, params, np.exp(params),
                   self.conf_int(exp=False), self.conf_int()]
        ##Shorten table (two tables?)
        CI = SimpleTable(CI, headers=['variable','parameter','exp(param)',
                                      'lower 95 CI', 'upper 95 CI',
                                      'lower 95 CI (exp)', 'upper 95 CI (exp)'
                                      ], title="Confidence Intervals")
        tests = np.array([self.wald_test(), self.score_test(),
                         self.likelihood_ratio_test()])
        tests = np.c_[np.array(['wald', 'score', 'likelihood ratio']),
                      tests, stats.chi2.sf(tests, len(params))]
        tests = SimpleTable(tests, headers=['test', 'test stat', 'p-value'],
                            title="Tests for Global Null")
        print(coeffs)
        print(CI)
        print(tests)
        #TODO: make print into return

    @cache_readonly
    def baseline(self):
        """
        estimate the baseline survival function

        Parameters
        ----------
        return_times : bool
            indicator of whether times should also be returned

        Returns
        -------
        baseline : ndarray
            array of predicted baseline survival probabilities
            at the observed times. If return_times is true, then
            an array whose first column is the times, and whose
            second column is the vaseline survival associated with that
            time

        """

        ##As function of t?
        ##Save baseline after first use? and check in other methods
        ##with hasattr?
        #TODO: do we need return_times argument?

        model = self.model
        baseline = KaplanMeier(model.surv).fit()
        return baseline.survival

    def predict(self, X, t):
        """
        estimate the hazard with a given vector of covariates

        Parameters
        ----------
        X : array-like
            matrix of covariate vectors. If t='all', must be
            only a single vector, or 'all'. If 'all' predict
            with the entire design matrix.
        t : non-negative int or "all"
            time(s) at which to predict. If t="all", then
            predict at all the observed times

        Returns
        -------
        probs : ndarray
            array of predicted survival probabilities

        """
        #TODO: for consistency move to models with params as argument
        #defaults ?
        ##As function of t?
        ##t='all' and matrix?
        ##t= arbitrary array of times?
        ##Remove coerce_0_1


        if X == 'all':
            X = self.model.exog
            times = self.model.times
            tind = np.unique(times)
            times = np.bincount(times)
            times = times[tind]
            baseline = self.baseline(t != 'all')
            baseline = np.repeat(baseline, times, axis=0)
        else:
            #TODO: rett not defined
            baseline = self.baseline(rett)
        if t == 'all':
            return -np.log(baseline) * np.exp(np.dot(X, self.params))
        else:
            return (-np.log(baseline[baseline[:,0] <= t][-1][0])
                    * np.exp(np.dot(X, self.params)))

    def plot(self, vector='mean', CI_band=False):
        """
        Plot the estimated survival curve for a given covariate vector

        Parameters
        ----------
        vector : array-like or 'mean'
            A vector of covariates. vector='mean' will use the mean
            vector
        CI_band : bool
            If true, then confidence bands for the survival curve are also
            plotted
        coerce_0_1 : bool
            If true, then the values for the survival curve be coerced to fit
            in the interval [0,1]

        Notes
        -----
        TODO: bring into new format with ax ? options, extras in plot

        """

        ##Add CI bands
        ##Adjust CI bands for coeff variance
        ##Update with predict

        if vector == 'mean':
            vector = self.exog_mean
        model = self.model
        km = KaplanMeier(model.surv)
        km = km.fit()
        km.results[0][0] = self.predict(vector, 'all')
        km.plot()

    def plot_baseline(self, CI_band=False):
        """
        Plot the estimated baseline survival curve

        Parameters
        ----------
        vector : array-like or 'mean'
            A vector of covariates. vector='mean' will use the mean
            vector
        CI_band : bool
            If true, then confidence bands for the survival curve are also
            plotted.

        Notes
        -----
        TODO: bring into new format with ax ? options, extras in plot

        """

        baseline = KaplanMeier(self.model.surv)
        baseline = baseline.fit()
        baseline.plot(CI_band)

    def baseline_object(self):
        """
        Get the KaplanMeier object that represents the baseline survival
        function

        Returns
        -------
        mod : KaplanMeier instance

        """

        return KaplanMeier(self.model.surv)

    def test_coefficients(self):
        """
        test whether the coefficients for each exogenous variable
        are significantly different from zero

        Returns
        -------
        res : ndarray
            An array, where each row represents a coefficient.
            The first column is the coefficient, the second is
            the standard error of the coefficient, the third
            is the z-score, and the fourth is the p-value.

        """

        params = self.params
        model = self.model
        ##Other methods (e.g. score?)
        ##if method == "wald":
        se = np.sqrt(np.diagonal(model.covariance(params)))
        z = params/se
        return np.c_[params,se,z,2 * stats.norm.sf(np.abs(z), 0, 1)]

    def wald_test(self, restricted=None):
        """
        Calculate the wald statistic for a hypothesis test
        against the global null

        Parameters
        ----------
        restricted : None or array_like
            values of the parameter under the Null hypothesis. If restricted
            is None, then the starting values are uses for the Null.

        Returns
        -------
        stat : float
            test statistic

        TODO: add pvalue, what's the distribution?

        """

        if restricted is None:
            #TODO: using start_params as alternative, restriction looks fragile
            restricted = self.model.start_params
        params = self.params
        model = self.model
        return np.dot((np.dot(params - restricted, model.information(params)))
                      , params - restricted)

    def score_test(self, restricted=None):
        """
        Calculate the score statistic for a hypothesis test against the global
        null

        Parameters
        ----------
        restricted : None or array_like
            values of the parameter under the Null hypothesis. If restricted
            is None, then the starting values are uses for the Null.

        Returns
        -------
        stat : float
            test statistic


        TODO: add pvalue, what's the distribution?

        """

        if restricted is None:
            restricted = self.model.start_params
        model = self.model
        score = model.score(restricted)
        cov = model.covariance(restricted)
        return np.dot(np.dot(score, cov), score)

    def likelihood_ratio_test(self, restricted=None):
        """
        Calculate the likelihood ratio for a hypothesis test against the global
        null

        Parameters
        ----------
        restricted : None or array_like
            values of the parameter under the Null hypothesis. If restricted
            is None, then the starting values are uses for the Null.

        Returns
        -------
        stat : float
            test statistic


        TODO: add pvalue, what's the distribution?

        """

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
        """
        Calculate confidence intervals for the model parameters

        Parameters
        ----------
        exp : logical value, indicating whether the confidence
            intervals for the exponentiated parameters

        see documentation for LikelihoodModel for other
        parameters

        Returns
        -------
        confint : ndarray
            An array, each row representing a parameter, where
            the first column gives the lower confidence limit
            and the second column gives the upper confidence
            limit

        """

        CI = super(CoxResults, self).conf_int(alpha, cols, method)
        if exp:
            CI = np.exp(CI)
        return CI

    def diagnostics(self):

        """
        initialized diagnostics for a fitted Cox model

        This attaches some diagnostic statistics to this instance

        TODO: replace with lazy cached attributes

        """

        ##Other residuals
        ##Plots
        ##Tests

        model = self.model
        censoring = model.censoring
        hazard = self.predict('all','all')
        mart = censoring - hazard
        self.martingale_resid = mart
        self.deviance_resid = (np.sign(mart) *
                               np.sqrt(2 * (-mart - censoring *
                                            np.log(censoring - mart))))
        self.phat = 1 - np.exp(-hazard)
        ind = censoring != 0
        exog = model.exog
        events = exog[ind]
        event_times = np.unique(model.times[ind])
        residuals = np.empty((1,len(self.params)))
        for i in range(len(event_times)):
            t = event_times[i]
            phat = 1 - np.exp(-self.predict('all',t))
            ind = event_times <= t
            self.phat = phat
            self.test = np.dot(phat[ind],exog[ind])
            self.test2 = events[i]
            residuals = np.r_[residuals,events[i] -
                              np.dot(phat[ind],exog[ind])[:,np.newaxis]]
        self.schoenfeld_resid = residuals[1:,:]
        print("diagnostics initialized")

    ##For plots, add spline
    def martingale_plot(self, covariate):
        """
        Plot the martingale residuals against a covariate
        (Must call diagnostics method first)

        Parameters
        ----------
        covariate : int
            index of the covariate to be plotted

        Notes
        -----
        do

        plt.show()

        To display a plot with the covariate values on the
        horizontal axis, and the martingale residuals for each
        observation on the vertical axis

        TODO: bring into new format with ax ? options, extras in plot

        """

        plt.plot(self.model.exog[:,covariate], self.martingale_resid,
                 marker='o', linestyle='None')

    def deviance_plot(self):
        """
        plot an index plot of the deviance residuals
        (must call diagnostics method first)

        Notes
        -----

        do

        plt.show()

        To display a plot with the index of the observation on the
        horizontal axis, and the deviance residuals for each
        observation on the vertical axis

        TODO: bring into new format with ax ? options, extras in plot

        """

        dev = self.deviance_resid
        plt.plot(np.arange(1,len(dev)+1), dev, marker='o', linestyle='None')

    def scheonfeld_plot(self):
        #TODO: not implemented yet
        pass


if __name__ == "__main__":
    import pandas
    #http://stat.ethz.ch/education/semesters/ss2011/seminar/contents/presentation_2.pdf
    dta = pandas.DataFrame(
                    [(6, 1, 1), (6, 1, 1), (6, 1, 1), (7, 1, 1), (10, 1, 1),
                     (13, 1, 1), (16, 1, 1), (22, 1, 1), (23, 1, 1),
                     (6, 0, 1), (9, 0, 1), (10, 0, 1), (11, 0, 1), (17, 0, 1),
                     (19, 0, 1), (20, 0, 1), (25, 0, 1), (32, 0, 1),
                     (32, 0, 1), (34, 0, 1), (35, 0, 1), (1, 1, 2), (1, 1, 2),
                     (2, 1, 2), (2, 1, 2), (3, 1, 2), (4, 1, 2), (4, 1, 2),
                     (5, 1, 2), (5, 1, 2), (8, 1, 2), (8, 1, 2), (8, 1, 2),
                     (8, 1, 2), (11, 1, 2), (11, 1, 2), (12, 1, 2),
                     (12, 1, 2), (15, 1, 2), (17, 1, 2), (22, 1, 2),
                     (23, 1, 2)],
                    columns=["duration", "status", "treatment"])

    #dta = pandas.read_csv("/home/skipper/scratch/remission.csv")
    surv = Survival("duration", event="status", data=dta)


    mod = KaplanMeier(surv, groups=dta["treatment"])
    res = mod.fit()

    # from lecture handout. times to finish a test in different noise
    # conditions. test is taken up at 12 minutes no matter what
    #www-personal.umich.edu/~yili/lect3notes.pdf

    dta2 = pandas.DataFrame([(9,1,1), (9.5,1,1), (9,1,1), (8.5,1,1), (10,1,1),
                            (10.5,1,1), (10,1,2), (12,1,2), (12,0,2),
                            (11,1,2), (12,1,2), (10.5,1,2), (12,1,3),
                            (12,0,3), (12,0,3), (12,0,3),
                            (12,0,3), (12,0,3)],
                            columns=["time","event","noise"])

    mod2 = KaplanMeier(Survival("time", event="event", data=dta2),
            groups=dta2["noise"])
    res2 = mod2.fit()
