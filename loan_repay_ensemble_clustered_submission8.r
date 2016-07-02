# This module contains the clustered ensemble of prediction models for Kaggle competion to
# predict potential credit risk per
# https://inclass.kaggle.com/c/adatelemz-si-platformok-2016-2-gyakorlat/
#
# Author: George Vyshnya
# Date: June 27-Jul 1, 2016
#
# Summary: 
# - Example of weighted ensemble prediction and submission demonstrated
# - xgboost, SVM linear kernel, RF, adaboost and NNET models with tuned parameters used in ensemble composition
# - Clustering of training data by Age groups applied

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

#modeler function for SVM linear
svmLinearModeler <- function(train, test, formula2verify, cost_factor=1) {
  print(paste0("Running svmLinearModeler"))
  svmMod6 <- svm(formula2verify, data=train, kernel="linear", probability=TRUE, cost=cost_factor)
  svmTestPred <- predict(svmMod6, newdata=test,probability=TRUE)
  probabilities <- attr(svmTestPred, "probabilities")
  head(probabilities)
  return (probabilities[,2])
}

#modeler function for SVM sigmoid
svmSigmoidModeler <- function(train, test, formula2verify, cost_factor=1,gamma=1,coef0=0) {
  print(paste0("Running svmSigmoidModeler"))
  svmMod6 <- svm(formula2verify, data=train, kernel="sigmoid", probability=TRUE, 
        cost=cost_factor, gamma=gamma, coef0=coef0)
  svmTestPred <- predict(svmMod6, newdata=test,probability=TRUE)
  probabilities <- attr(svmTestPred, "probabilities")
  head(probabilities)
  return (probabilities[,2])
}

#modeler function for RF
rfModeler <- function(train, test, formula2verify, alpha=2601) {
  print(paste0("Running rfModeler"))
  rfMod6 <- randomForest(formula2verify, data=train, ntree=alpha)
  rfTestPred <- predict(rfMod6, newdata=test,type="prob")[,2]
  head(rfTestPred)
  return (rfTestPred)
}

#modeler function for adaboost 
adaboostModeler <- function(train, test, formula2verify, mfinal=25, maxdepth=3, minsplit = 15) {
  print(paste0("Running adaboostModeler"))
  
  control <- rpart.control(cp = 0.01,xval = 0,maxdepth=maxdepth, minsplit=minsplit)
  mod6 <- ada(formula2verify, data = train, max.iter=20, iter = mfinal, bag.frac = 0.1, nu = 0.5,
       model.coef=TRUE,bag.shift=FALSE,delta=10^(-10),
	   loss=c("exponential"),
       type=c("gentle"),   # real  
	   control = control)

  pred <- predict(mod6,newdata=test, type="probs")
  
  probabilities <- pred
  head(probabilities)
  return (probabilities[,2])
}

#modeler function for nnet 
nnetModeler <- function(train, test, formula2verify, mfinal=25) {
  print(paste0("Running nnetModeler"))
  
  mod6 <- nnet(formula2verify, data = train, size = 2, rang = 0.1,
               decay = 5e-4, maxit = mfinal)
  
  pred <- predict(mod6,newdata=test, type="raw")
  
  probabilities <- pred
  head(probabilities)
  return (probabilities)
}

#modeler function for xgboost 
xgboostModeler <- function(train, test, formula2verify, mfinal=4, maxdepth=2) {
  print(paste0("Running xgboostModeler"))
  
  fitControl <- trainControl(classProbs = TRUE, summaryFunction = twoClassSummary, method="cv", number=mfinal)

  xgbGrid <-  expand.grid(max_depth = maxdepth,
                        nrounds = (1:5)*50,
                        eta = c(0.1,0.2,0.3),
                        gamma = 0,
                        colsample_bytree = .70,
                        min_child_weight = 1 )

  modelFit <- train(formula2verify,
               data = train,
               method = 'xgbTree',
               metric = 'ROC',
               trControl = fitControl,
               tuneGrid = xgbGrid)



  pred <- predict(modelFit, newdata = test, type="prob")
  
  probabilities <- pred
  head(probabilities)
  return (probabilities[,2])
}

# submission output file name
output.csv.file <- "Submission_ensemble8_ageclustered.csv"
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

# cols to omit from the imputation and prediction set:
all <- select(all, -cat_dem, -cat_zip1, -cat_zip2, -cat_zip3, -cat_zip4, -cat_occupid, 
        -cat_pinconf, -cat_bsresid, -num_sameplace, -month, -year, -num_ndep, -cat_opstate)

str(all)
customer_id <- all$customer_id

# start imputation
print(paste("Do imputation of pre-processed data",Sys.time()))
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

# convert character, integer and some numeric fields to factors
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

# proportion of target class labels before SMOTE
prop.table(table(train$target))

# SMOTE target of the train class to make classes more balanced
# SMOTE = Synthetic Minority Over-sampling Technique.
COMMON_NO <- round(nrow(train)*0.7357401)
RARE_NO <- nrow(train) - COMMON_NO
Over <-( (0.6 * COMMON_NO) - RARE_NO ) / RARE_NO
Under <- (0.4 * COMMON_NO) / (RARE_NO * Over)
Over_Perc <- round(Over, 1) * 100
Under_Perc <- round(Under, 1) * 100
train <- SMOTE(target ~ ., train, k=3, perc.over = Over_Perc, perc.under=Under_Perc) #correct unbalanced target

# proportion of target class labels after SMOTE
prop.table(table(train$target))

# cluster data by group age

#first remove customer_id from training set
train <- select(train, -customer_id)
train1 <- subset(train, AgeGroup == 1)
train2 <- subset(train, AgeGroup == 2)
train3 <- subset(train, AgeGroup == 3)
train4 <- subset(train, AgeGroup == 4 | AgeGroup == 5)

# remove AgeGroup from the clustered training sets
train1 <- select(train1, -AgeGroup)
train2 <- select(train2, -AgeGroup)
train3 <- select(train3, -AgeGroup)
train4 <- select(train4, -AgeGroup)

test1 <- subset(test, AgeGroup == 1)
test2 <- subset(test, AgeGroup == 2)
test3 <- subset(test, AgeGroup == 3)
test4 <- subset(test, AgeGroup == 4 | AgeGroup == 5)

# store customer id vectors for each of the test set's clusters
customer_id1 <- test1$customer_id
customer_id2 <- test2$customer_id
customer_id3 <- test3$customer_id
customer_id4 <- test4$customer_id

#remove customer_id and AgeGroup from clustered data sets
test1 <- select(test1, -customer_id, -AgeGroup)
test2 <- select(test2, -customer_id, -AgeGroup)
test3 <- select(test3, -customer_id, -AgeGroup)
test4 <- select(test4, -customer_id, -AgeGroup)

# start model training
set.seed(825)
print(paste("Do model training",Sys.time()))
formula <- as.formula(target ~ .)




#fit model by Age clusters and make predictions
# start making predictions
print(paste("Fit models and make predictions",Sys.time()))
	
# predict with SVM, Linear kernel
set.seed(825)
p1_1 <- svmLinearModeler (train1, test1, formula, cost_factor=0.1) #,gamma=1,coef0=0.004)
p2_1 <- svmLinearModeler (train2, test2, formula, cost_factor=0.1) #,gamma=1,coef0=0.004)
p3_1 <- svmLinearModeler (train3, test3, formula, cost_factor=0.1) #,gamma=1,coef0=0.004)
p4_1 <- svmLinearModeler (train4, test4, formula, cost_factor=0.1) #,gamma=1,coef0=0.004)

# predict with RF
set.seed(825)
p1_2 <- rfModeler (train1, test1, formula, alpha=1501)
p2_2 <- rfModeler (train2, test2, formula, alpha=1501)
p3_2 <- rfModeler (train3, test3, formula, alpha=1501) 
p4_2 <- rfModeler (train4, test4, formula, alpha=1501) 

# predict with adaboost implementation from ada
set.seed(825)
p1_3 <- adaboostModeler (train1, test1, formula, mfinal=35, maxdepth=5, minsplit = 15)
p2_3 <- adaboostModeler (train2, test2, formula, mfinal=35, maxdepth=5, minsplit = 15)
p3_3 <- adaboostModeler (train3, test3, formula, mfinal=35, maxdepth=5, minsplit = 15)
p4_3 <- adaboostModeler (train4, test4, formula, mfinal=35, maxdepth=5, minsplit = 15)

# predict with nnet
set.seed(825)
p1_4 <- nnetModeler (train1, test1, formula, mfinal=50)
p2_4 <- nnetModeler (train2, test2, formula, mfinal=50)
p3_4 <- nnetModeler (train3, test3, formula, mfinal=50)
p4_4 <- nnetModeler (train4, test4, formula, mfinal=50)

# predict with xgboost
set.seed(825)
p1_5 <- xgboostModeler (train1, test1, formula, mfinal=4, maxdepth=2)
p2_5 <- xgboostModeler (train2, test2, formula, mfinal=4, maxdepth=2)
p3_5 <- xgboostModeler (train3, test3, formula, mfinal=4, maxdepth=2)
p4_5 <- xgboostModeler (train4, test4, formula, mfinal=4, maxdepth=2)

n.models <- 10

prediction1 <- (2*p1_1 + 3*p1_2 + p1_3 + p1_4 + 3*p1_5)/n.models 
prediction2 <- (2*p2_1 + 3*p2_2 + p2_3 + p2_4 + 3*p2_5)/n.models 
prediction3 <- (2*p3_1 + 3*p3_2 + p3_3 + p3_4 + 3*p3_5)/n.models 
prediction4 <- (2*p4_1 + 3*p4_2 + p4_3 + p4_4 + 3*p4_5)/n.models 


# Output submission
threshold <- 0.5
print(paste("Output",Sys.time()))
PredTestLabels1 <- as.factor(ifelse(prediction1 > threshold, "1.0", "0.0"))
PredTestLabels2 <- as.factor(ifelse(prediction2 > threshold, "1.0", "0.0"))
PredTestLabels3 <- as.factor(ifelse(prediction3 > threshold, "1.0", "0.0"))
PredTestLabels4 <- as.factor(ifelse(prediction4 > threshold, "1.0", "0.0"))
df1 <- data.frame(customer_id = customer_id1, target = PredTestLabels1)
df2 <- data.frame(customer_id = customer_id2, target = PredTestLabels2)
df3 <- data.frame(customer_id = customer_id3, target = PredTestLabels3)
df4 <- data.frame(customer_id = customer_id4, target = PredTestLabels4)
MySubmission <- rbind(df1,df2,df3,df4)
write.csv(MySubmission, output.csv.file, row.names=FALSE)

print(paste("Elapsed time: ",Sys.time()-strt))

# That's all!
########################################

