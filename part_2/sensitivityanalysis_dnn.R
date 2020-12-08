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

### Impute missing values
impute <- preProcess(x_train, method = c('center','scale','medianImpute'))
x_train <- predict(impute, x_train)
x_test <- predict(impute, x_test)

### Train a DNN model
bst <- keras_model_sequential()
bst %>%
  layer_dense(units = 30,  activation = 'relu', input_shape = ncol(x_train)) %>%
  layer_dense(units = 30, activation = 'relu') %>%
  layer_dense(units = 30, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'sigmoid') %>%
  compile(
    loss = 'binary_crossentropy',
    optimizer = optimizer_rmsprop(lr = 0.001),
    metrics = c('accuracy')
  )
bst %>% fit(x_train, y_train, epochs = 5, batch_size = 128)

### Verify that test AUC is same as what's reported in the paper
bst_pred_test <- as.vector(predict(bst, x_test))
roc(y_test, bst_pred_test)
ci.auc(roc(y_test, bst_pred_test), conf.level = 0.95)

### Sensitivity analysis from here and below
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
      ival0 <- x_test_df[which(x_test_df[, col_orig] < 0)[1], col_orig]
      ival1 <- x_test_df[which(x_test_df[, col_orig] > 0)[1], col_orig]
      jval0 <- x_test_df[which(x_test_df[, col_new] < 0)[1], col_new]
      jval1 <- x_test_df[which(x_test_df[, col_new] > 0)[1], col_new]
      pred_test_orig <- predict(bst, x_test[x_test_df[, col_orig]==ival1,])
      x_test_new <- x_test[x_test_df[, col_orig]==ival1,]
      x_test_new[,which(colnames(x_test_df)==col_orig)] <- jval0
      x_test_new[,which(colnames(x_test_df)==col_new)] <- jval1
      pred_test_new <- predict(bst, x_test_new)
      res[i,j] <- mean(pred_test_new - pred_test_orig)
    }
  }
}
res
write.csv(res, paste0("./Analysis/ESI_dnn.csv"), row.names=TRUE)

# Demographic variables:
# age, gender, ethnicity, race, lang, insurance_status

# Gender
col_orig <- "gender.Female"; col_new <- "gender.Male"
col_orig <- "gender.Male"; col_new <- "gender.Female"
ival0 <- x_test_df[which(x_test_df[, col_orig] < 0)[1], col_orig]
ival1 <- x_test_df[which(x_test_df[, col_orig] > 0)[1], col_orig]
jval0 <- x_test_df[which(x_test_df[, col_new] < 0)[1], col_new]
jval1 <- x_test_df[which(x_test_df[, col_new] > 0)[1], col_new]
table(x_test_df[,col_orig])
table(x_test_df[,col_new])
pred_test_orig <- predict(bst, x_test[x_test_df[,col_orig]==ival1,])
x_test_new <- x_test[x_test_df[, col_orig]==ival1,]
x_test_new[,which(colnames(x_test_df)==col_orig)] <- ival0
x_test_new[,which(colnames(x_test_df)==col_new)] <- jval1
pred_test_new <- predict(bst, x_test_new)
mean(pred_test_new - pred_test_orig)

# Ethnicity
col_orig <- "ethnicity.Hispanic.or.Latino"; col_new <- "ethnicity.Non.Hispanic"
col_orig <- "ethnicity.Non.Hispanic"; col_new <- "ethnicity.Hispanic.or.Latino"
ival0 <- x_test_df[which(x_test_df[, col_orig] < 0)[1], col_orig]
ival1 <- x_test_df[which(x_test_df[, col_orig] > 0)[1], col_orig]
jval0 <- x_test_df[which(x_test_df[, col_new] < 0)[1], col_new]
jval1 <- x_test_df[which(x_test_df[, col_new] > 0)[1], col_new]
table(x_test_df[,col_orig])
table(x_test_df[,col_new])
pred_test_orig <- predict(bst, x_test[x_test_df[,col_orig]==ival1,])
x_test_new <- x_test[x_test_df[, col_orig]==ival1,]
x_test_new[,which(colnames(x_test_df)==col_orig)] <- ival0
x_test_new[,which(colnames(x_test_df)==col_new)] <- jval1
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
    ival0 <- x_test_df[which(x_test_df[, col_orig] < 0)[1], col_orig]
    ival1 <- x_test_df[which(x_test_df[, col_orig] > 0)[1], col_orig]
    jval0 <- x_test_df[which(x_test_df[, col_new] < 0)[1], col_new]
    jval1 <- x_test_df[which(x_test_df[, col_new] > 0)[1], col_new]
    if (i != j) {
      pred_test_orig <- predict(bst, x_test[x_test_df[, col_orig]==ival1,])
      x_test_new <- x_test[x_test_df[, col_orig]==ival1,]
      x_test_new[,which(colnames(x_test_df)==col_orig)] <- ival0
      x_test_new[,which(colnames(x_test_df)==col_new)] <- jval1
      pred_test_new <- predict(bst, x_test_new)
      res[i,j] <- mean(pred_test_new - pred_test_orig)
    }
  }
}
res
colnames(res) <- set
rownames(res) <- set
write.csv(res, paste0("./Analysis/race_dnn.csv"), row.names=TRUE)

# Language
col_orig <- "lang.English"; col_new <- "lang.Other"
col_new <- "lang.English"; col_orig <- "lang.Other"
ival0 <- x_test_df[which(x_test_df[, col_orig] < 0)[1], col_orig]
ival1 <- x_test_df[which(x_test_df[, col_orig] > 0)[1], col_orig]
jval0 <- x_test_df[which(x_test_df[, col_new] < 0)[1], col_new]
jval1 <- x_test_df[which(x_test_df[, col_new] > 0)[1], col_new]
table(x_test_df[,col_orig])
table(x_test_df[,col_new])
pred_test_orig <- predict(bst, x_test[x_test_df[,col_orig]==ival1,])
x_test_new <- x_test[x_test_df[, col_orig]==ival1,]
x_test_new[,which(colnames(x_test_df)==col_orig)] <- ival0
x_test_new[,which(colnames(x_test_df)==col_new)] <- jval1
pred_test_new <- predict(bst, x_test_new)
mean(pred_test_new - pred_test_orig)

# Age
col <- "age"
table(x_test_df[,col])
pred_test_orig <- predict(bst, x_test)
pdiff <- c()
coldiff <- seq(-1.5, 2.7, by=0.1)
minval <- min(x_test_df[,col], na.rm=TRUE)
maxval <- max(x_test_df[,col], na.rm=TRUE)
for (diff in coldiff) {
  x_test_new <- x_test
  newage <- x_test_new[,which(colnames(x_test_df)==col)] + diff
  newage[newage < minval] <- minval
  newage[newage > maxval] <- maxval
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
    ival0 <- x_test_df[which(x_test_df[, col_orig] < 0)[1], col_orig]
    ival1 <- x_test_df[which(x_test_df[, col_orig] > 0)[1], col_orig]
    jval0 <- x_test_df[which(x_test_df[, col_new] < 0)[1], col_new]
    jval1 <- x_test_df[which(x_test_df[, col_new] > 0)[1], col_new]
    if (i != j) {
      pred_test_orig <- predict(bst, x_test[x_test_df[, col_orig]==ival1,])
      x_test_new <- x_test[x_test_df[, col_orig]==ival1,]
      x_test_new[,which(colnames(x_test_df)==col_orig)] <- ival0
      x_test_new[,which(colnames(x_test_df)==col_new)] <- jval1
      pred_test_new <- predict(bst, x_test_new)
      res[i,j] <- mean(pred_test_new - pred_test_orig)
    }
  }
}
res
colnames(res) <- set
rownames(res) <- set
write.csv(res, paste0("./Analysis/insurance_dnn.csv"), row.names=TRUE)

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
    ival0 <- x_test_df[which(x_test_df[, col_orig] < 0)[1], col_orig]
    ival1 <- x_test_df[which(x_test_df[, col_orig] > 0)[1], col_orig]
    jval0 <- x_test_df[which(x_test_df[, col_new] < 0)[1], col_new]
    jval1 <- x_test_df[which(x_test_df[, col_new] > 0)[1], col_new]
    if (i != j) {
      pred_test_orig <- predict(bst, x_test[x_test_df[, col_orig]==ival1,])
      x_test_new <- x_test[x_test_df[, col_orig]==ival1,]
      x_test_new[,which(colnames(x_test_df)==col_orig)] <- ival0
      x_test_new[,which(colnames(x_test_df)==col_new)] <- jval1
      pred_test_new <- predict(bst, x_test_new)
      res[i,j] <- mean(pred_test_new - pred_test_orig)
    }
  }
}
res
colnames(res) <- set
rownames(res) <- set
write.csv(res, paste0("./Analysis/previousdispo_dnn.csv"), row.names=TRUE)

# Number of ED visits
col <- "n_edvisits"
table(x_test_df[,col])
pred_test_orig <- predict(bst, x_test)
pdiff <- c()
coldiff <- seq(-100, 100, by=10)
center <- 3.657767
sd <- 10.65898
coldiff <- (coldiff-center)/sd
# coldiff <- seq(-0.35, 35, by=0.5)
minval <- min(x_test_df[,col], na.rm=TRUE)
maxval <- max(x_test_df[,col], na.rm=TRUE)
for (diff in coldiff) {
  x_test_new <- x_test
  newcol <- x_test_new[,which(colnames(x_test_df)==col)] + diff
  newcol[newcol < minval] <- minval
  newcol[newcol > maxval] <- maxval
  x_test_new[,which(colnames(x_test_df)==col)] <- newcol
  pred_test_new <- predict(bst, x_test_new)
  pdiff <- c(pdiff, mean(pred_test_new - pred_test_orig))
}
plot(coldiff, pdiff)
plot(coldiff, pdiff, main="DNN", xlab="Perturbation", ylab="Probability change",
     ylim=c(-0.2, 0.2))
abline(h=0)

# Number of admissions
col <- "n_admissions"
table(x_test_df[,col])
pred_test_orig <- predict(bst, x_test)
pdiff <- c()
coldiff <- seq(-38, 20, by=1)
minval <- min(x_test_df[,col], na.rm=TRUE)
maxval <- max(x_test_df[,col], na.rm=TRUE)
for (diff in coldiff) {
  x_test_new <- x_test
  newcol <- x_test_new[,which(colnames(x_test_df)==col)] + diff
  newcol[newcol < minval] <- minval
  newcol[newcol > maxval] <- maxval
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
minval <- min(x_test_df[,col], na.rm=TRUE)
maxval <- max(x_test_df[,col], na.rm=TRUE)
for (diff in coldiff) {
  x_test_new <- x_test
  newcol <- x_test_new[,which(colnames(x_test_df)==col)] + diff
  newcol[newcol < minval] <- minval
  newcol[newcol > maxval] <- maxval
  x_test_new[,which(colnames(x_test_df)==col)] <- newcol
  pred_test_new <- predict(bst, x_test_new)
  pdiff <- c(pdiff, mean(pred_test_new - pred_test_orig))
}
plot(coldiff, pdiff)

