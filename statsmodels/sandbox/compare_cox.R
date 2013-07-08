Rossi <- read.csv("http://www.ams.jhu.edu/~dan/550.400/datasets/recidivism/Rossi%20data.csv")
S<-Surv(time=Rossi$week,event=Rossi$arrest)
SF<-survfit(S~1)
SF<-survfit(Surv(week,arrest)~1,data=Rossi)

library(car)
Rossi$age.cat <- recode(Rossi$age, " lo:19=1; 20:25=2; 26:30=3; 31:hi=4 ")

#plot(SF)
cph <- coxph(S~age, data=Rossi, model=TRUE, x=TRUE, y=TRUE, method="breslow")
cph.2 <- coxph(Surv(week, arrest) ~ fin + prio, data=Rossi)
cph.strat <- coxph(Surv(week, arrest) ~ fin + prio + strata(age.cat), data=Rossi)


coxph(S~race,data=Rossi)
coxph(S~fin,data=Rossi)
coxph(S~mar,data=Rossi)
coxph(S~prio,data=Rossi)
coxph(S~paro,data=Rossi)
CPH<-coxph(S~age+race+fin+wexp+prio+paro,data=Rossi)

coxph(Surv(week,arrest)~age,data=Rossi)

c.fit <- coxph(Surv(aml1$time, aml1$status)~1, method="efron", model=TRUE)
coxph.fit <- survfit(c.fit)

c.fit.all <- coxph(Surv(aml$time, aml$status)~aml$x, model=TRUE)
coxph.fit.all <- survfit(c.fit.all)
