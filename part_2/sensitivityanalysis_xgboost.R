
library(readr)
library(dplyr)
library(reshape2)
library(parallel)
library(caret)
library(xgboost)
library(doMC)
library(pROC)
library(keras)
registerDoMC(5) # For parallelization


### Load processed data
# 5v_sparseMatrix.RData and 5v_indeces_list.RData can be found in sensitivity_analysis.zip
# We got them by running the original authors' build_boost.R script.
load('./Results/5v_sparseMatrix.RData')
load('./Results/5v_indeces_list.RData')
indeces <- indeces_list[[1]]
x <- as.matrix(dataset$x)
y <- dataset$y
rm(dataset)
x_test <- x[indeces$i_test,]
y_test <- y[indeces$i_test]
x_train <- x[-indeces$i_test,]
y_train <- y[-indeces$i_test]
rm(x); rm(y)

### Train a XGBoost model
bst <- xgboost(data = x_train, label = y_train,
               max_depth = 20, eta = 0.3,
               nthread = 5, nrounds = 30,
               eval_metric = 'auc',
               objective = "binary:logistic",
               colsample_bylevel = 0.05)
save(bst, file="./Results/final_boost_model.Rdata")

### Verify that test AUC is same as what's reported in the paper
bst_pred_test <- predict(bst, x_test)
roc(y_test, bst_pred_test) # 0.926
ci.auc(roc(y_test, bst_pred_test), conf.level = 0.95) # 95% CI: 0.9237-0.9283

### Sensitivity analysis from here and below

# Make a data frame version of the test data
x_test_df <- data.frame(x_test)

# ESI variables: 1 (resuscitation), 2 (emergent), 3 (urgent), 4 (less urgent), 5 (non-urgent)
table(x_test_df$esi.1)
table(x_test_df$esi.2)
table(x_test_df$esi.3)
table(x_test_df$esi.4)
table(x_test_df$esi.5)
res <- matrix(0, ncol=5, nrow=5)
for (i in c(1:5)) {
  for (j in c(1:5)) {
    col_orig <- paste0("esi.", i)
    col_new <- paste0("esi.", j)
    if (i != j) {
      pred_test_orig <- predict(bst, x_test[x_test_df[, col_orig]==1,])
      x_test_new <- x_test[x_test_df[, col_orig]==1,]
      x_test_new[,which(colnames(x_test_df)==col_orig)] <- 0
      x_test_new[,which(colnames(x_test_df)==col_new)] <- 1
      pred_test_new <- predict(bst, x_test_new)
      res[i,j] <- mean(pred_test_new - pred_test_orig)
    }
  }
}
res
write.csv(res, paste0("./Analysis/ESI_boost.csv"), row.names=TRUE)

# Demographic variables:
# age, gender, ethnicity, race, lang, insurance_status

# Gender
col_orig <- "gender.Female"; col_new <- "gender.Male"
col_orig <- "gender.Male"; col_new <- "gender.Female"
table(x_test_df[,col_orig])
table(x_test_df[,col_new])
pred_test_orig <- predict(bst, x_test[x_test_df[,col_orig]==1,])
x_test_new <- x_test[x_test_df[, col_orig]==1,]
x_test_new[,which(colnames(x_test_df)==col_orig)] <- 0
x_test_new[,which(colnames(x_test_df)==col_new)] <- 1
pred_test_new <- predict(bst, x_test_new)
mean(pred_test_new - pred_test_orig)

# Ethnicity
col_orig <- "ethnicity.Hispanic.or.Latino"; col_new <- "ethnicity.Non.Hispanic"
col_orig <- "ethnicity.Non.Hispanic"; col_new <- "ethnicity.Hispanic.or.Latino"
table(x_test_df[,col_orig])
table(x_test_df[,col_new])
pred_test_orig <- predict(bst, x_test[x_test_df[,col_orig]==1,])
x_test_new <- x_test[x_test_df[, col_orig]==1,]
x_test_new[,which(colnames(x_test_df)==col_orig)] <- 0
x_test_new[,which(colnames(x_test_df)==col_new)] <- 1
pred_test_new <- predict(bst, x_test_new)
mean(pred_test_new - pred_test_orig)

# Race
table(x_test_df$race.American.Indian.or.Alaska.Native)
table(x_test_df$race.Asian)
table(x_test_df$race.Black.or.African.American)
table(x_test_df$race.Native.Hawaiian.or.Other.Pacific.Islander)
table(x_test_df$race.White.or.Caucasian)
set <- c("race.American.Indian.or.Alaska.Native", "race.Asian",
         "race.Black.or.African.American", "race.Native.Hawaiian.or.Other.Pacific.Islander",
         "race.White.or.Caucasian")
res <- matrix(0, ncol=5, nrow=5)
for (i in c(1:5)) {
  for (j in c(1:5)) {
    col_orig <- set[i]
    col_new <- set[j]
    if (i != j) {
      pred_test_orig <- predict(bst, x_test[x_test_df[, col_orig]==1,])
      x_test_new <- x_test[x_test_df[, col_orig]==1,]
      x_test_new[,which(colnames(x_test_df)==col_orig)] <- 0
      x_test_new[,which(colnames(x_test_df)==col_new)] <- 1
      pred_test_new <- predict(bst, x_test_new)
      res[i,j] <- mean(pred_test_new - pred_test_orig)
    }
  }
}
res
colnames(res) <- set
rownames(res) <- set
write.csv(res, paste0("./Analysis/race_boost.csv"), row.names=TRUE)

# Language
col_orig <- "lang.English"; col_new <- "lang.Other"
col_new <- "lang.English"; col_orig <- "lang.Other"
table(x_test_df[,col_orig])
table(x_test_df[,col_new])
pred_test_orig <- predict(bst, x_test[x_test_df[,col_orig]==1,])
x_test_new <- x_test[x_test_df[, col_orig]==1,]
x_test_new[,which(colnames(x_test_df)==col_orig)] <- 0
x_test_new[,which(colnames(x_test_df)==col_new)] <- 1
pred_test_new <- predict(bst, x_test_new)
mean(pred_test_new - pred_test_orig)

# Age
col <- "age"
table(x_test_df[,col])
pred_test_orig <- predict(bst, x_test)
pdiff <- c()
coldiff <- seq(-50, 50, by=5)
for (diff in coldiff) {
  x_test_new <- x_test
  newage <- x_test_new[,which(colnames(x_test_df)==col)] + diff
  newage[newage < 18] <- 18
  newage[newage > 105] <- 105
  x_test_new[,which(colnames(x_test_df)==col)] <- newage
  pred_test_new <- predict(bst, x_test_new)
  pdiff <- c(pdiff, mean(pred_test_new - pred_test_orig))
}
plot(coldiff, pdiff)

# Insurance
table(x_test_df$insurance_status.Commercial)
table(x_test_df$insurance_status.Medicaid)
table(x_test_df$insurance_status.Medicare)
table(x_test_df$insurance_status.Other)
table(x_test_df$insurance_status.Self.pay)
set <- c("insurance_status.Commercial", "insurance_status.Medicaid",
         "insurance_status.Medicare", "insurance_status.Other",
         "insurance_status.Self.pay")
res <- matrix(0, ncol=5, nrow=5)
for (i in c(1:5)) {
  for (j in c(1:5)) {
    col_orig <- set[i]
    col_new <- set[j]
    if (i != j) {
      pred_test_orig <- predict(bst, x_test[x_test_df[, col_orig]==1,])
      x_test_new <- x_test[x_test_df[, col_orig]==1,]
      x_test_new[,which(colnames(x_test_df)==col_orig)] <- 0
      x_test_new[,which(colnames(x_test_df)==col_new)] <- 1
      pred_test_new <- predict(bst, x_test_new)
      res[i,j] <- mean(pred_test_new - pred_test_orig)
    }
  }
}
res
colnames(res) <- set
rownames(res) <- set
write.csv(res, paste0("./Analysis/insurance_boost.csv"), row.names=TRUE)

# Hospital use: previousdispo, n_edvisits, n_admissions, n_surgeries
# previousdispo.Admit and n_admissions are in top 20 important variables

# Previous disposition
table(x_test_df$previousdispo.Admit)
table(x_test_df$previousdispo.AMA)
table(x_test_df$previousdispo.Discharge)
table(x_test_df$previousdispo.Eloped)
set <- c("previousdispo.Admit", "previousdispo.AMA",
         "previousdispo.Discharge", "previousdispo.Eloped")
res <- matrix(0, ncol=4, nrow=4)
for (i in c(1:4)) {
  for (j in c(1:4)) {
    col_orig <- set[i]
    col_new <- set[j]
    if (i != j) {
      pred_test_orig <- predict(bst, x_test[x_test_df[, col_orig]==1,])
      x_test_new <- x_test[x_test_df[, col_orig]==1,]
      x_test_new[,which(colnames(x_test_df)==col_orig)] <- 0
      x_test_new[,which(colnames(x_test_df)==col_new)] <- 1
      pred_test_new <- predict(bst, x_test_new)
      res[i,j] <- mean(pred_test_new - pred_test_orig)
    }
  }
}
res
colnames(res) <- set
rownames(res) <- set
write.csv(res, paste0("./Analysis/previousdispo_boost.csv"), row.names=TRUE)

# Number of ED visits
col <- "n_edvisits"
table(x_test_df[,col])
hist(x_test_df[, col])
mean(x_test_df[, col] < 10)
pred_test_orig <- predict(bst, x_test)
pdiff <- c()
coldiff <- seq(-100, 100, by=10)
for (diff in coldiff) {
  x_test_new <- x_test
  newcol <- x_test_new[,which(colnames(x_test_df)==col)] + diff
  newcol[newcol < 0] <- 0
  newcol[newcol > 374] <- 374
  x_test_new[,which(colnames(x_test_df)==col)] <- newcol
  pred_test_new <- predict(bst, x_test_new)
  pdiff <- c(pdiff, mean(pred_test_new - pred_test_orig))
}
plot(coldiff, pdiff, main="XGBoost", xlab="Perturbation", ylab="Probability change",
     ylim=c(-0.2, 0.2))
abline(h=0)

# Number of admissions
col <- "n_admissions"
table(x_test_df[,col])
pred_test_orig <- predict(bst, x_test)
pdiff <- c()
coldiff <- seq(-10, 10, by=1)
for (diff in coldiff) {
  x_test_new <- x_test
  newcol <- x_test_new[,which(colnames(x_test_df)==col)] + diff
  newcol[newcol < 0] <- 0
  newcol[newcol > 48] <- 48
  x_test_new[,which(colnames(x_test_df)==col)] <- newcol
  pred_test_new <- predict(bst, x_test_new)
  pdiff <- c(pdiff, mean(pred_test_new - pred_test_orig))
}
plot(coldiff, pdiff)

# Number of surgeries
col <- "n_surgeries"
table(x_test_df[,col])
pred_test_orig <- predict(bst, x_test)
pdiff <- c()
coldiff <- seq(-10, 10, by=1)
for (diff in coldiff) {
  x_test_new <- x_test
  newcol <- x_test_new[,which(colnames(x_test_df)==col)] + diff
  newcol[newcol < 0] <- 0
  newcol[newcol > 37] <- 37
  x_test_new[,which(colnames(x_test_df)==col)] <- newcol
  pred_test_new <- predict(bst, x_test_new)
  pdiff <- c(pdiff, mean(pred_test_new - pred_test_orig))
}
plot(coldiff, pdiff)

