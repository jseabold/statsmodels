from statsmodels.sandbox.phreg import PHreg
from statsmodels.datasets import ovarian_cancer

dta = ovarian_cancer.load()

# this version doesn't use the Survival object
# it has its own internal format

mod = PHreg(dta.endog.time.values, dta.endog.event.values, dta.exog,
            ties='breslow')
res = mod.fit()


from statsmodels.sandbox.survival2 import CoxPH, Survival
# using built-in example data
mod_ss = CoxPH(dta.endog, dta.exog, ties='breslow')
res_ss = mod_ss.fit()

# assume you don't have a Survial object in dta.endog already
surv = Survival(time1=data.data['time'], event=data['event'])
mod_ss = CoxPH(surv, dta.exog, ties='breslow')
res_ss = mod_ss.fit()
