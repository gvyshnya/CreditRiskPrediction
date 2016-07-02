# This module contains the reference cross-validation analysis for
# RF prediction model for Kaggle competion to
# predict possible default on loans per
# https://inclass.kaggle.com/c/adatelemz-si-platformok-2016-2-gyakorlat/
#
# Author: George Vyshnya
# Date: June 27-Jul 1, 2016
#
# Summary
# - custom implementation of k-fold cross-validation with SMOTE transformation of train data
# - the clustering of data by K-means algorithm used in this example (with 5 classes constructed)

library(caret)
library(caTools)
library(plyr)
library(dplyr)
library(car)
library(Matrix)
library(mice)
library(e1071)
library(vcd)
library(xgboost)
library(nnet)
library(gbm)
library(rpart)
library(rpart.plot)
library(randomForest)
library(ROCR)
library(ranger)
library(glmnet)
library(ggplot2)
library(pROC)
library(DMwR)
library(adabag)
library(ada)
library(flexclust)

Age2AgeGroup <- function (x) {
	result <- 1 # under 25
	age <- x
	if (age < 25) {result <- 1}
	if (age >= 25 & age < 36) {result <- 2}
	if (age >= 36 & age < 49) {result <- 3}
	if (age >= 49 & age < 64) {result <- 4}
	if (age >= 64 ) {result <- 5}
	result
}

Income2IncomeGroup <- function (x) {
	result <- 1 # under 50000
	income <- x
	if (income < 50000) {result <- 1}
	if (income >= 50000 & income < 75000) {result <- 2}
	if (income >= 75000 & income < 100000) {result <- 3}
	if (income >= 100000 & income < 125000) {result <- 4}
	if (income >= 125000 & income < 150000) {result <- 5}
	if (income >= 150000 & income < 175000) {result <- 6}
	if (income >= 175000 & income < 200000) {result <- 7}
	if (income >= 200000 ) {result <- 8}
	result

}

crossValidate <- function(df, formula2verify, nfolds, modeler,alpha=0,theshold=0.5) {
  cv.acc <- vector(mode="numeric", length=nfolds)
  set.seed(113341)
  folds <- sample(rep(1:nfolds,length=nrow(df)))
  for(kk in 1:nfolds) {
	df.train <- df[folds!=kk,]
	df.test <- df[folds==kk,]
	df.train <- SMOTE(formula2verify, df.train, k=3, perc.over = 100, perc.under=200) #correct unbalanced target
    pred <- modeler(train=df.train, test=df.test, formula2verify=formula2verify, alpha=alpha)
    tab <- table(df.test$target, pred > theshold)
    cv.acc[kk] <- sum(diag(tab))/sum(tab)
    print(paste0("Finished fold ",kk,"/",nfolds, "; ntrees=",alpha,
	      "; accuracy for this fold: ", cv.acc[kk]))
  } 
  avgAcc <- mean(cv.acc)
  return (avgAcc)
}

#modeler function
rfModeler <- function(train, test, formula2verify, alpha=2601) {
  print(paste0("Running rfModeler"))
  rfMod6 <- randomForest(formula2verify, data=train, ntree=alpha)
  rfTestPred <- predict(rfMod6, newdata=test,type="prob")[,2]
  head(rfTestPred)
  return (rfTestPred)
}


############################################
# Executable part of the script
############################################

# submission output file name
output.csv.file <- "Submission_rf2_ageclustered.csv"
strt<-Sys.time()

# read data
print(paste("Load data",Sys.time()))
train <- read.csv("training.csv", stringsAsFactors = FALSE) #, na.strings=c("NA","NaN", "")
test <- read.csv("testing.csv", stringsAsFactors = FALSE)
originalTest <- test

str(train)
str(test)

# do some data exploration
print(paste("Do data exploration",Sys.time()))
table(train$target)
#Which of the following variables has at least one missing observation?
sapply(train, function(x) sum(is.na(x)))
sapply(test, function(x) sum(is.na(x)))

table(train$target, train$cat_pinconf)

table(train$cat_pinconf)

# do data preprocessing
print(paste("Do data pre-processing",Sys.time()))
trainTargets <- train$target
trainLimited <- select(train, -target)

# combine test and training data into one data set for ease of manipulation
all <- rbind(trainLimited,test)
end_trn <- nrow(trainLimited)
end <- nrow(all)
end_trn
end

# eliminate some noicy or misleading columns from the imputation and prediction set:
all <- select(all, -cat_dem, -cat_zip1, -cat_zip2, -cat_zip3, -cat_zip4, -cat_occupid, 
        -cat_pinconf, -cat_bsresid, -num_sameplace, -num_ndep, -cat_opstate)

str(all)
customer_id <- all$customer_id

# start imputation
print(paste("Do imputation of pre-processed data and feature engineering", Sys.time()))

# NA in training:  f_odprot , f_pnid, f_ppce, f_ppoa, f_ptaxid
# NA in testing:  f_odprot, f_pnid, f_ppce, f_ppoa, f_ptaxid
# cat_occupid , cat_pinconf have some NA, too but hey are excluded from predictors

# impute NA using median or most common value
all$f_odprot[is.na(all$f_odprot)] <- 0
all$f_pnid[is.na(all$f_pnid)] <- 1
all$f_ppce[is.na(all$f_ppce)] <- 1
all$f_ppoa[is.na(all$f_ppoa)] <- 0
all$f_ptaxid[is.na(all$f_ptaxid)] <- 1

all$customer_id  <- customer_id
str(all)

# add a new factor column for age group
all$AgeGroup <- as.factor(sapply(all$num_age, function (x) Age2AgeGroup(x) ))
# remove num_age column
all <- select(all, -num_age)
# add a new factor column for income group
all$IncomeGroup <- as.factor(sapply(all$num_income, function (x) Income2IncomeGroup(x) ))
# remove num_income column
all <- select(all, -num_income)

#add a new column for elapsed time
#current.year <- as.integer(format(Sys.Date(), "%Y")
#current.month <- as.integer(format(Sys.Date(), "%M")
# all$TimeSince <- as.integer(12*(current.year - all$year)+(current.month-all$month))

# remove year and month
all <- select(all, -month, -year)

# do clustering via K-means technique
print(paste("Do data clustering",Sys.time()))
# note we do not need target var in the limitedTrain, to facilitate appropriate clustering
limitedTrain <- all[1:end_trn,]    
limitedTest <- all[(end_trn+1):end,]

# remove customer_id from the limited sets
limitedTrain <- select(limitedTrain, -customer_id)
limitedTest <- select(limitedTest, -customer_id)
# convert factors to numeric values
limitedTrain$IncomeGroup <- as.numeric(limitedTrain$IncomeGroup)
limitedTest$IncomeGroup <- as.numeric(limitedTest$IncomeGroup)
limitedTrain$AgeGroup <- as.numeric(limitedTrain$AgeGroup)
limitedTest$AgeGroup <- as.numeric(limitedTest$AgeGroup)

preproc <- preProcess(limitedTrain)
normTrain <- predict(preproc, limitedTrain)
normTest <- predict(preproc, limitedTest)

set.seed(825)
Kmeans <- kmeans(normTrain, centers = 5)
Cluster <- Kmeans$cluster
table(Cluster)

km.kcca <- as.kcca(Kmeans, normTrain)
clusterTrain <- predict(km.kcca)
clusterTest <- predict(km.kcca, newdata=normTest)
table(clusterTest)


# convert character, integer and some numeric fields to factors
print(paste("Convert vars to factors, do log transformation",Sys.time()))
all$f_brsame <- as.factor(all$f_brsame)
all$f_fname <- as.factor(all$f_fname)
all$f_cap <- as.factor(all$f_cap)
all$f_instantpur <- as.factor(all$f_instantpur)
all$f_landline <- as.factor(all$f_landline)
all$f_mname <- as.factor(all$f_mname)
all$f_odprot <- as.factor(all$f_odprot)
all$f_othercard <- as.factor(all$f_othercard)
all$f_pnid <- as.factor(all$f_pnid)
all$f_ppce <- as.factor(all$f_ppce)
all$f_ppoa <- as.factor(all$f_ppoa)
all$f_ptaxid <- as.factor(all$f_ptaxid)
all$f_sex <- as.factor(all$f_sex)
all$f_tel <- as.factor(all$f_tel)
all$f_wscity <- as.factor(all$f_wscity)
all$f_wsstate <- as.factor(all$f_wsstate)
all$cat_civstatus <- as.factor(all$cat_civstatus)
all$cat_residtype <- as.factor(all$cat_residtype)
all$num_sameplace_1 <- as.factor(all$num_sameplace_1)
all$has_persref <- as.factor(all$has_persref)
all$num_account <- as.factor(all$num_account)
all$num_samejob <- log(all$num_samejob+1) # do a log transformation


# put updated values to train and test data set
train <- all[1:end_trn,]
train$target <- trainTargets

# two lines below are needed to elinimate issues with class outcome since we will use classProbs = TRUE
# see more details in the thread per http://stackoverflow.com/questions/18402016/error-when-i-try-to-predict-class-probabilities-in-r-caret
levels <- unique(train$target) 
train$target <- factor(train$target, labels=make.names(levels))
str(train)

test <- all[(end_trn+1):end,]
str(test)

# SMOTE target of the train class to make classes more balanced
# SMOTE = Synthetic Minority Over-sampling Technique.
# train <- SMOTE(target ~ ., train, perc.over = 100, perc.under=200)

# cluster train and test data into the kmean-identified 5-class cluster

train1 <- subset(train, clusterTrain == 1)
train2 <- subset(train, clusterTrain == 2)
train3 <- subset(train, clusterTrain == 3)
train4 <- subset(train, clusterTrain == 4)
train5 <- subset(train, clusterTrain == 5)

test1 <- subset(test, clusterTest == 1)
test2 <- subset(test, clusterTest == 2)
test3 <- subset(test, clusterTest == 3)
test4 <- subset(test, clusterTest == 4)
test5 <- subset(test, clusterTest == 5)

# store customer id vectors for each of the test set's clusters
customer_id1 <- test1$customer_id
customer_id2 <- test2$customer_id
customer_id3 <- test3$customer_id
customer_id4 <- test4$customer_id
customer_id5 <- test5$customer_id

# eliminate customer id from trainin and test sets
test1 <- select(test1, -customer_id)
test2 <- select(test2, -customer_id)
test3 <- select(test3, -customer_id)
test4 <- select(test4, -customer_id)
test5 <- select(test5, -customer_id)

train1 <- select(train1, -customer_id)
train2 <- select(train2, -customer_id)
train3 <- select(train3, -customer_id)
train4 <- select(train4, -customer_id)
train5 <- select(train5, -customer_id)

# start model training
set.seed(825)
formula <- as.formula(target ~ .)
max.trees <- 2601
print(paste("Do model cross-validation - cluster #1",Sys.time()))

alpha <- 40
rfTestAccuracy1 <- crossValidate(df=train1, formula2verify=formula, 
		nfolds=5,alpha=alpha, modeler=rfModeler, theshold=0.5)
print(paste("Accuracy set 1, RF, trees - ",alpha, ": ", rfTestAccuracy1))
alpha <- 50
rfTestAccuracy1 <- crossValidate(df=train1, formula2verify=formula, 
		nfolds=5,alpha=alpha, modeler=rfModeler, theshold=0.5)
print(paste("Accuracy set 1, RF, trees - ",alpha, ": ", rfTestAccuracy1))
alpha <- 75
rfTestAccuracy1 <- crossValidate(df=train1, formula2verify=formula, 
		nfolds=5,alpha=alpha, modeler=rfModeler, theshold=0.5)
print(paste("Accuracy set 1, RF, trees - ",alpha, ": ", rfTestAccuracy1))
alpha <- 100
rfTestAccuracy1 <- crossValidate(df=train1, formula2verify=formula, 
		nfolds=5,alpha=alpha, modeler=rfModeler, theshold=0.5)
print(paste("Accuracy set 1, RF, trees - ",alpha, ": ", rfTestAccuracy1))
alpha <- 150
rfTestAccuracy1 <- crossValidate(df=train1, formula2verify=formula, 
		nfolds=5,alpha=alpha, modeler=rfModeler, theshold=0.5)
print(paste("Accuracy set 1, RF, trees - ",alpha, ": ", rfTestAccuracy1))

alpha <- 250
rfTestAccuracy1 <- crossValidate(df=train1, formula2verify=formula, 
		nfolds=5,alpha=alpha, modeler=rfModeler, theshold=0.5)
print(paste("Accuracy set 1, RF, trees - ",alpha, ": ", rfTestAccuracy1))

alpha <- 350
rfTestAccuracy1 <- crossValidate(df=train1, formula2verify=formula, 
		nfolds=5,alpha=alpha, modeler=rfModeler, theshold=0.5)
print(paste("Accuracy set 1, RF, trees - ",alpha, ": ", rfTestAccuracy1))

alpha <- 500
rfTestAccuracy1 <- crossValidate(df=train1, formula2verify=formula, 
		nfolds=5,alpha=alpha, modeler=rfModeler, theshold=0.5)
print(paste("Accuracy set 1, RF, trees - ",alpha, ": ", rfTestAccuracy1))

print(paste("Do model cross-validation - cluster #2",Sys.time()))
alpha <- 50
rfTestAccuracy2 <- crossValidate(df=train2, formula2verify=formula, 
		nfolds=5,alpha=alpha, modeler=rfModeler, theshold=0.5)
print(paste("Accuracy set 2, RF, trees - ",alpha, ": ", rfTestAccuracy2))

print(paste("Do model cross-validation - cluster #3",Sys.time()))
alpha <- 50
rfTestAccuracy3 <- crossValidate(df=train3, formula2verify=formula, 
		nfolds=5,alpha=alpha, modeler=rfModeler, theshold=0.5)
print(paste("Accuracy set 3, RF, trees - ",alpha, ": ", rfTestAccuracy3))

print(paste("Do model cross-validation - cluster #4",Sys.time()))
alpha <- 50
rfTestAccuracy4 <- crossValidate(df=train4, formula2verify=formula, 
		nfolds=5,alpha=alpha, modeler=rfModeler, theshold=0.5)
print(paste("Accuracy set 4, RF, trees - ",alpha, ": ", rfTestAccuracy4))

print(paste("Do model cross-validation - cluster #5",Sys.time()))
alpha <- 50
rfTestAccuracy5 <- crossValidate(df=train5, formula2verify=formula, 
		nfolds=5,alpha=alpha, modeler=rfModeler, theshold=0.5)
print(paste("Accuracy set 4, RF, trees - ",alpha, ": ", rfTestAccuracy5))

##########################################################
print(paste("Elapsed time: ",Sys.time()-strt))

# That's all!
########################################

