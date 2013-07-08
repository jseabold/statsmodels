dta <- read.csv("/home/skipper/statsmodels/statsmodels-skipper/statsmodels/datasets/strikes/strikes.csv")
library(R2nparray)
library(survival)

surv <- Surv(dta$duration)

km <- survfit(surv~1, type="kaplan-meier")

km.results.nogroups.nocensor <- list(event=km$n.event, 
                                     time=km$time, 
                                     nrisk=km$n.risk,
                                     censored=km$n.censor, 
                                     survival=km$surv,
                                     std_err_cumhazard=km$std.err, 
                                     std_err=summary(km)$std.err, 
                                     lower=km$lower, 
                                     upper=km$upper)

sink(file="km_results.py")
cat("           # No Groups No Censor\n\n")
sink()

R2nparray(km.results.nogroups.nocensor, fname="./km_results.py", 
          append=TRUE)

bins <- unique(dta$iprod)
bins <- sort(bins)
groups <- cut(dta$iprod, bins, include.lowest=TRUE)

km.nocensor.group <- survfit(surv~groups, type="kaplan-meier")

km.results.groups.nocensor <- list(event=km.nocensor.group$n.event, 
                                   time=km.nocensor.group$time, 
                                   nrisk=km.nocensor.group$n.risk,
                                   censored=km.nocensor.group$n.censor, 
                                   survival=km.nocensor.group$surv,
                                   std_err_cumhazard=km.nocensor.group$std.err, 
                                   std_err=summary(km.nocensor.group)$std.err, 
                                   lower=km.nocensor.group$lower, 
                                   upper=km.nocensor.group$upper)

sink(file="km_results.py", append=TRUE)
cat("           # Groups No Censor\n\n")
sink()

R2nparray(km.results.groups.nocensor, fname="./km_results.py",
          append=TRUE)


# artificially censor
event = rep(1, length(dta$duration))
event[dta$duration > 80] = 0

surv <- Surv(dta$duration, event)

km.censor.nogroup <- survfit(surv~1, type="kaplan-meier")

km.results.nogroups.censor <- list(event=km.censor.nogroup$n.event, 
                                   time=km.censor.nogroup$time, 
                                   nrisk=km.censor.nogroup$n.risk,
                                   censored=km.censor.nogroup$n.censor, 
                                   survival=km.censor.nogroup$surv,
                                   std_err_cumhazard=km.censor.nogroup$std.err, 
                                   std_err=summary(km.censor.nogroup)$std.err, 
                                   lower=km.censor.nogroup$lower, 
                                   upper=km.censor.nogroup$upper)


sink(file="km_results.py", append=TRUE)
cat("           # No Groups Censor\n\n")
sink()

R2nparray(km.results.nogroups.censor, fname="./km_results.py",
          append=TRUE)

km.censor.group <- survfit(surv~groups, type="kaplan-meier")

km.results.groups.censor <- list(event=km.censor.group$n.event, 
                                   time=km.censor.group$time, 
                                   nrisk=km.censor.group$n.risk,
                                   censored=km.censor.group$n.censor, 
                                   survival=km.censor.group$surv,
                                   std_err_cumhazard=km.censor.group$std.err, 
                                   std_err=summary(km.censor.group)$std.err, 
                                   lower=km.censor.group$lower, 
                                   upper=km.censor.group$upper)


sink(file="km_results.py", append=TRUE)
cat("           # Groups Censor\n\n")
sink()

R2nparray(km.results.groups.censor, fname="./km_results.py",
          append=TRUE)
