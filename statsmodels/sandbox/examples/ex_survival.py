import pandas
import statsmodels.api as sm

aml = sm.datasets.get_rdataset("leukemia", "survival", cache=True).data

# recode for clarity
aml.rename(columns=dict(x="group", status="event"), inplace=True)

# Drop the censored observations
grp =  aml.drop(aml[aml["event"]==0].index).groupby("group")
grp["time"].agg(dict(mean=np.mean, median=np.median)).T

# Treat censored observations as exact
aml.groupby("group")["time"].agg(dict(mean=np.mean, median=np.median)).T

# True mean and median is biased (likely underestimated due to the censoring)
from survival2 import Survival, KaplanMeier

# first let's look at the Maintained
surv = Survival("time", event="event",
                data=aml[aml["group"]=="Maintained"])


km_fit = KaplanMeier(surv).fit()

