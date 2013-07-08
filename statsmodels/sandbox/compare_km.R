library(survival)
aml1 <- aml[aml$x == "Maintained", ]
km.fit <- survfit(Surv(aml1$time, aml1$status)~1, type="kaplan-meier",
                  data=aml1)

aml2 <- aml[aml$x == "Nonmaintained", ]
km.fit.2 <- survfit(Surv(aml2$time, aml2$status)~1, type="kaplan-meier",
                    data=aml2)

plot(km.fit)
km.fit.all <- survfit(Surv(aml$time, aml$status)~x, type="kaplan-meier",
                  data=aml)

res <- data.frame(time=km.fit.all$time, n.risk=km.fit.all$n.risk,
                  n.event=km.fit.all$n.event,                   
                  n.censor=km.fit.all$n.censor)
#aml$y = "HUH"
#aml$y[1:5] = "OH"
#km.fit.all <- survfit(Surv(aml$time, aml$status)~x+y, type="kaplan-meier",
#                  data=aml)
#plot(km.fit.all, lty = 2:3) 
#legend(100, .9, c("Maintenance", "No Maintenance"), lty = 2:3) 
