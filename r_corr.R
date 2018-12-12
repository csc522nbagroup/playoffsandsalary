install.packages("corrplot")
install.packages("readxl")
library(ggplot2)
library(corrplot)
library("readxl")

#Function to normalize data
n = function(x) {(x - min(x)) / (max(x) - min(x))}

#function to produce residual plot
plot_pred_resid = function(lm, pcol = "gray", lcol = "blue") {
  plot(fitted(lm), resid(lm),
       col = pcol, pch = 20, cex = 1.5,
       main = "Predicted vs. Residuals",
       xlab = "Predicted", ylab = "Residuals")
  abline(h = 0, col = lcol, lwd = 2)
}

#####################
# not log transformed
#####################
data = df

#correlations = cor(data)
#corrplot(correlations)

#normalize predictor variables
X = lapply(data[,-2], n)
y=data[2]
#combine x and y
data = cbind(y,X)

#original model
lm1 <- lm(Salary~., data=data)
AIC(lm1)
summary(lm1)$adj.r.squared
paste("RMSE: ", sqrt(mean(lm1$residuals^2)))
#plot_pred_resid(lm = lm1)
#par(mfrow=c(2,2))
#plot(lm1)

#####################
# log - transformed
#####################
data = df
X = lapply(data[,-2], n)
y = log(data[2])

log_data = cbind(y,X)

lm2 <- lm(Salary~., data=log_data)

y_pred <- fitted(lm2)
y_pred <- exp(y_pred)
y_act <- exp(log_data['Salary'])
sq_residuals <- (y_pred - y_act)^2
rmse <- sqrt(mean(sq_residuals))
paste("RMSE: ", rmse)
tss <- sum((y_act-mean(y_act[['Salary']]))^2)
rss <- sum(sq_residuals)
rsquared <- 1 - (rss/tss)
adj_rsquared = 1 - (1 - rsquared) * (nrow(log_data) - 1) / (nrow(log_data) - ncol(log_data) - 1) 
paste("Adj-R-squared: ", adj_rsquared)


lm3 <- lm(Salary ~ Year + Age + MPG + VORP + DRB + TOPG + Yrs_Exp +  MPG+ I(MPG^2) + PTS + PTS:I(MPG^2), data = log_data)

#plot_pred_resid(lm = lm3)
y_pred <- fitted(lm3)
y_pred <- exp(y_pred)
y_act <- exp(log_data['Salary'])
sq_residuals <- (y_pred - y_act)^2
rmse <- sqrt(mean(sq_residuals))
paste("RMSE: ", rmse)
tss <- sum((y_act-mean(y_act[['Salary']]))^2)
rss <- sum(sq_residuals)
rsquared <- 1 - (rss/tss)
adj_rsquared = 1 - (1 - rsquared) * (nrow(log_data) - 1) / (nrow(log_data) - ncol(log_data) - 1) 
paste("Adj-R-squared: ", adj_rsquared)



lm4 <- lm(Salary ~ Year + Age + MPG + Yrs_Exp + PTS:MPG + PER:MPG + PER:PTS + VORP + DRB + TOPG + PTS + Yrs_Exp + Age:Yrs_Exp, data=log_data)

#plot_pred_resid(lm = lm4)
y_pred <- fitted(lm4)
y_pred <- exp(y_pred)
y_act <- exp(log_data['Salary'])
sq_residuals <- (y_pred - y_act)^2
rmse <- sqrt(mean(sq_residuals))
paste("RMSE: ", rmse)
tss <- sum((y_act-mean(y_act[['Salary']]))^2)
rss <- sum(sq_residuals)
rsquared <- 1 - (rss/tss)
adj_rsquared = 1 - (1 - rsquared) * (nrow(log_data) - 1) / (nrow(log_data) - ncol(log_data) - 1) 
paste("Adj-R-squared: ", adj_rsquared)


lm5 <- lm(Salary ~ Year + MPG + Height + Weight + PTS:MPG + USG_PCT:PER + Age:Yrs_Exp + PER:MPG + PER:PTS + TS_PCT + STL + BLK + AST + DRB + TOPG + PTS + Age + Yrs_Exp, data=log_data)

#plot_pred_resid(lm = lm4)
y_pred <- fitted(lm5)
y_pred <- exp(y_pred)
y_act <- exp(log_data['Salary'])
sq_residuals <- (y_pred - y_act)^2
rmse <- sqrt(mean(sq_residuals))
paste("RMSE: ", rmse)
tss <- sum((y_act-mean(y_act[['Salary']]))^2)
rss <- sum(sq_residuals)
rsquared <- 1 - (rss/tss)
adj_rsquared = 1 - (1 - rsquared) * (nrow(log_data)-1) / (nrow(log_data) - ncol(log_data) - 1) 
paste("Adj-R-squared: ", adj_rsquared)
#summary(lm5)

lm6 <- lm(Salary ~ Year + PER:VORP + PER:DRB + PER:TOPG + MPG + PTS + Age + Yrs_Exp + TS_PCT+ PER:MPG + PTS:Age + AST + Age:Yrs_Exp, data=log_data)

#plot_pred_resid(lm = lm6)
y_pred <- fitted(lm6)
y_pred <- exp(y_pred)
y_act <- exp(log_data['Salary'])
sq_residuals <- (y_pred - y_act)^2
rmse <- sqrt(mean(sq_residuals))
paste("RMSE: ", rmse)
tss <- sum((y_act-mean(y_act[['Salary']]))^2)
rss <- sum(sq_residuals)
rsquared <- 1 - (rss/tss)
adj_rsquared = 1 - (1 - rsquared) * (nrow(log_data)-1) / (nrow(log_data) - ncol(log_data) - 1) 
paste("Adj-R-squared: ", adj_rsquared)
#par(mfrow=c(2,2))
#plot(lm5)

