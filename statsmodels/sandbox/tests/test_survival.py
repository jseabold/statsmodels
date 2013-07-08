'''Tests for Survival models: KaplanMeier and CoxPH

currently just smoke tests to check which attributes and methods raise
exceptions
'''


import numpy as np
import pandas
from statsmodels.sandbox.survival2 import Survival, KaplanMeier, CoxPH

from numpy.testing import assert_equal


from statsmodels.datasets import ovarian_cancer

dta = ovarian_cancer.load()
darray = np.asarray(dta['data'])

from results.km_results import KaplanMeierResults

class CheckCoxPH(object):

    def test_smoke_model(self):
        #smoke test to check status and for refactoring
        model = self.model
        results = self.results

        #Note: not available (exception): '_data', 'endog_names', 'exog_names'

        results.baseline()
        results.baseline_object()
        results.conf_int()
        results.cov_params()
        #results.deviance_plot()
        results.diagnostics()
        results.f_test(np.ones(results.params.shape))
        #results.initialize()   #internal
        results.likelihood_ratio_test()
        #results.load()     #inherited pickle load
        results.martingale_plot(1)
        #results.plot()  #BUG: possibly
        results.plot_baseline()
        #results.predict()   #check arguments
        #results.remove_data()
        #results.save()    #inherited pickle save
        results.scheonfeld_plot()
        results.score_test()
        results.summary()
        results.t()
        results.t_test(np.eye(results.params.shape[0]))
        results.test_coefficients()
        results.wald_test()

        assert_equal(results.params.shape, (results.model.exog.shape[1],))
        results.normalized_cov_params
        results.scale
        results.phat
        results.exog_mean
        results.test
        results.test2
        results.params
        results.names
        results.deviance_resid
        results.martingale_resid
        results._cache
        results.model
        results._data_attr
        results.schoenfeld_resid

        results.wald_test()

        #model.initialize() #internal
        #model.fit()  #already called
        model._hessian_proc(results.params)
        model._loglike_proc(results.params)
        model._score_proc(results.params)
        #model._stratify_func() #arguments ?  internal
        model.confint_dist()
        model.covariance(results.params)

        model.hessian(results.params)
        model.information(results.params)

        model.loglike(results.params)
        model.score(results.params)

        #model.predict()
        #results.predict(model.exog[-2:], model.times[-2:])  #BUG
        assert_equal(results.predict('all', 'all').shape, results.model.times.shape)

        model.stratify(1)

        model._str_censoring
        model.df_resid
        model._str_times
        model.confint_dist
        model.d
        model._str_exog
        model._str_d
        model.strata
        model.exog_mean
        model.times
        model.surv
        if self.has_strata:
            model.strata_groups   #not in this example
        model.names
        model.ttype
        model.start_params
        model.ties
        model.censoring
        model.exog

class XestCoxPH1(CheckCoxPH):

    @classmethod
    def setup_class(cls):
        exog = darray[:,range(2,6)]
        surv = Survival(0, censoring=1, data=darray)
        cls.model = model = CoxPH(surv, exog)
        cls.results = model.fit()
        cls.has_strata = False

class XestCoxPHStrata(CheckCoxPH):

    @classmethod
    def setup_class(cls):
        exog = darray[:,range(2,6)]
        surv = Survival(0, censoring=1, data=darray)

        gene1 = exog[:,0]
        expressed = gene1 > gene1.mean()
        ##replacing the column for th first gene with the indicatore variable
        exog[:,0] = expressed
        cls.model = model = CoxPH(surv, exog)
        model.stratify(0,copy=False)
        cls.has_strata = True
        cls.results = model.fit()


class CheckKaplanMeier(object):

    def test_survival(self):
        np.testing.assert_almost_equal(self.res1.survival,
                                       self.res2.survival, 5)

    def test_nrisk(self):
        np.testing.assert_almost_equal(self.res1.model.nrisk,
                                       self.res2.nrisk, 5)

    def test_nevents(self):
        np.testing.assert_almost_equal(self.res1.model.event,
                                       self.res2.event, 5)

    def test_ntime(self):
        np.testing.assert_almost_equal(self.res1.model.time,
                                       self.res2.time, 5)

    def test_std_err_cumhazard(self):
        np.testing.assert_almost_equal(self.res1.std_err_cumhazard,
                                       self.res2.std_err_cumhazard, 5)

    def test_std_err(self):
        #we drop the censored cases in _plot_curve rather than not keeping
        # them around like summary.survfit in R, so
        std_err = self.res1.std_err
        std_err = std_err[self.res1.model.endog[:,1].astype(bool)]
        np.testing.assert_almost_equal(std_err,
                                       self.res2.std_err, 5)

    def test_diff(self):
        pass

    def test_plot(self):
        pass

    def test_conf_int(self):
        conf_int = self.res1.conf_int()
        res_conf_int = np.c_[self.res2.lower, self.res2.upper]
        np.testing.assert_almost_equal(conf_int,
                                       res_conf_int, 5)

    def xest_isolate_curve(self):
        #separate to get isolated failure
        model = self.res1.model
        results = self.res1
        if model.groups is not None:
            results.isolate_curve(model.groups[0])
        #TODO: test "exog", ["exog1", "exog2"], ["exog"], None

class TestKaplanMeierNoCensorNoGroup(CheckKaplanMeier):
    @classmethod
    def setup_class(cls):
        from statsmodels.datasets import strikes
        dta = strikes.load_pandas()
        surv = Survival("duration", event=None, data=dta.data)
        model = KaplanMeier(surv)
        cls.res1 = model.fit()
        cls.res2 = KaplanMeierResults(group=False, censor=False)

class TestKaplanMeierNoCensorGroup(CheckKaplanMeier):
    #no exog

    @classmethod
    def setup_class(cls):
        from statsmodels.datasets import strikes
        dta = strikes.load_pandas()
        bins = np.unique(dta.exog.values)
        bins.sort()
        #TODO: what version of pandas was cut added in?
        groups = pandas.cut(dta.exog.values.squeeze(), bins,
                            include_lowest=True, labels=False)

        surv = Survival("duration", data=dta.data)
        model = KaplanMeier(surv, groups=groups)
        cls.res1 = model.fit()
        cls.res2 = KaplanMeierResults(group=True, censor=False)

class TestKaplanMeierCensorNoGroup(CheckKaplanMeier):
    @classmethod
    def setupClass(cls):
        from statsmodels.datasets import strikes
        dta = strikes.load_pandas().data
        dta["event"] = 1
        dta.ix[dta["duration"] > 80, "event"] = 0
        surv = Survival("duration", event="event", data=dta)

        cls.res1 = KaplanMeier(surv).fit()
        cls.res2 = KaplanMeierResults(group=False, censor=True)


class TestKaplanMeierCensorGroup(CheckKaplanMeier):
    @classmethod
    def setup_class(cls):
        from statsmodels.datasets import strikes
        dta = strikes.load_pandas().data
        dta["event"] = 1
        dta.ix[dta["duration"] > 80, "event"] = 0
        surv = Survival("duration", event="event", data=dta)
        bins = np.unique(dta["iprod"].values)
        bins.sort()
        groups = pandas.cut(dta["iprod"], bins, include_lowest=True,
                            labels=False)
        model = KaplanMeier(surv, groups=groups)
        cls.res1 = model.fit()
        cls.res2 = KaplanMeierResults(group=True, censor=True)

if __name__ == '__main__':
    import nose
    #tt = TestCoxPH1()
    tt = TestKaplanMeier3()
    tt.setup_class()
    tt.test_smoke_model()
    nose.runmodule(argv=[__file__,'-s', '-v'], exit=False)
