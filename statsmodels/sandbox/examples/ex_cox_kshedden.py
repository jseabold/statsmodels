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
surv = Survival(time1=dta.data['time'], event=dta.data['event'])
mod_ss = CoxPH(surv, dta.exog, ties='breslow')
res_ss = mod_ss.fit()

# let's profile
import cProfile, pstats
params = res_ss.params
cProfile.run("mod_ss.hessian(params)", "ss.prof")

params = res.params
cProfile.run("mod.hessian(params)", "ks.prof")

p1 = pstats.Stats("ss.prof")
p1.strip_dirs().sort_stats('time').print_stats()

p2 = pstats.Stats("ks.prof")
p2.strip_dirs().sort_stats('time').print_stats()
