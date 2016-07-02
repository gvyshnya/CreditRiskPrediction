# This module contains the clustered ensemble of prediction models for Kaggle competion to
# predict credit risk of potential defaults on loans per
# https://inclass.kaggle.com/c/adatelemz-si-platformok-2016-2-gyakorlat/
#
# Author: George Vyshnya
# Date: June 27 - Jul 1, 2016
#
# Summary: 
# - mixed clustering by age and then k-means subclustering inside the youngster subpopulation
# - applied equal-weight ensemble of trained models (SVM linear, RF, xgboost, adaboost, nnet) with parameters from cv

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
library(mgcv)

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

#############################################
# Executable part of the script
#############################################

# submission output file name
output.csv.file <- "Submission_ensemble6_mixedcluster.csv"
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

# proportion of target class labels before SMOTE
prop.table(table(train$target))

nrow(train)

# SMOTE target of the train class to make classes more balanced
# SMOTE = Synthetic Minority Over-sampling Technique.
COMMON_NO <- round(nrow(train)*0.7357401)
RARE_NO <- nrow(train) - COMMON_NO
Over <-( (0.6 * COMMON_NO) - RARE_NO ) / RARE_NO
Under <- (0.4 * COMMON_NO) / (RARE_NO * Over)
Over_Perc <- round(Over, 1) * 100
Under_Perc <- round(Under, 1) * 100
train <- SMOTE(target ~ ., train, k=3, perc.over = Over_Perc, perc.under=Under_Perc) #correct unbalanced target via over-sampling

# proportion of target class labels after SMOTE
prop.table(table(train$target))

nrow(train)


# cluster data by age group, do additional k-mean clustering among younger population
# of age groups 1-2
print(paste("Do data clustering",Sys.time()))

#first remove customer_id from training set
train <- select(train, -customer_id)

train4 <- subset(train, AgeGroup == 3)
train5 <- subset(train, AgeGroup == 4 | AgeGroup == 5)

test4 <- subset(test, AgeGroup == 3)
test5 <- subset(test, AgeGroup == 4 | AgeGroup == 5)

train.youngsters <- subset(train, AgeGroup == 1 | AgeGroup == 2)
test.youngsters <- subset(test, AgeGroup == 1 | AgeGroup == 2)

# do clustering of younster subsets via K-means technique

# note we do not need target var in the limitedTrain, to facilitate appropriate clustering
limitedTrain <- train.youngsters    
limitedTest <- test.youngsters

# remove customer_id from the limited test
limitedTest <- select(limitedTest, -customer_id)
# remove target from the limisted train set
limitedTrain <- select(limitedTrain, -target)
# convert factors to numeric values to run clustering
limitedTrain$IncomeGroup <- as.numeric(limitedTrain$IncomeGroup)
limitedTest$IncomeGroup <- as.numeric(limitedTest$IncomeGroup)
limitedTrain$AgeGroup <- as.numeric(limitedTrain$AgeGroup)
limitedTest$AgeGroup <- as.numeric(limitedTest$AgeGroup)

limitedTrain$f_brsame <- as.numeric(limitedTrain$f_brsame)
limitedTrain$f_cap <- as.numeric(limitedTrain$f_cap)
limitedTrain$f_fname  <- as.numeric(limitedTrain$f_fname)      
limitedTrain$f_instantpur <- as.numeric(limitedTrain$f_instantpur)  
limitedTrain$f_landline <- as.numeric(limitedTrain$f_landline)  
limitedTrain$f_mname <- as.numeric(limitedTrain$f_mname) 
limitedTrain$f_odprot <- as.numeric(limitedTrain$f_odprot) 
limitedTrain$f_othercard <- as.numeric(limitedTrain$f_othercard) 
limitedTrain$f_pnid <- as.numeric(limitedTrain$f_pnid)       
limitedTrain$f_ppce <- as.numeric(limitedTrain$f_ppce)      
limitedTrain$f_ppoa <- as.numeric(limitedTrain$f_ppoa)          
limitedTrain$f_ptaxid <- as.numeric(limitedTrain$f_ptaxid)         
limitedTrain$f_sex <- as.numeric(limitedTrain$f_sex)        
limitedTrain$f_tel <- as.numeric(limitedTrain$f_tel)         
limitedTrain$f_wscity <- as.numeric(limitedTrain$f_wscity)         
limitedTrain$f_wsstate <- as.numeric(limitedTrain$f_wsstate)     
limitedTrain$has_persref <- as.numeric(limitedTrain$has_persref)     
limitedTrain$cat_civstatus <- as.numeric(limitedTrain$cat_civstatus)   
limitedTrain$cat_residtype <- as.numeric(limitedTrain$cat_residtype)    
limitedTrain$num_account <- as.numeric(limitedTrain$num_account)    
limitedTrain$num_samejob <- as.numeric(limitedTrain$num_samejob)       
limitedTrain$num_sameplace_1 <- as.numeric(limitedTrain$num_sameplace_1) 

limitedTest$f_brsame <- as.numeric(limitedTest$f_brsame)
limitedTest$f_cap <- as.numeric(limitedTest$f_cap)
limitedTest$f_fname  <- as.numeric(limitedTest$f_fname)      
limitedTest$f_instantpur <- as.numeric(limitedTest$f_instantpur)  
limitedTest$f_landline <- as.numeric(limitedTest$f_landline)  
limitedTest$f_mname <- as.numeric(limitedTest$f_mname) 
limitedTest$f_odprot <- as.numeric(limitedTest$f_odprot) 
limitedTest$f_othercard <- as.numeric(limitedTest$f_othercard) 
limitedTest$f_pnid <- as.numeric(limitedTest$f_pnid)       
limitedTest$f_ppce <- as.numeric(limitedTest$f_ppce)      
limitedTest$f_ppoa <- as.numeric(limitedTest$f_ppoa)          
limitedTest$f_ptaxid <- as.numeric(limitedTest$f_ptaxid)         
limitedTest$f_sex <- as.numeric(limitedTest$f_sex)        
limitedTest$f_tel <- as.numeric(limitedTest$f_tel)         
limitedTest$f_wscity <- as.numeric(limitedTest$f_wscity)         
limitedTest$f_wsstate <- as.numeric(limitedTest$f_wsstate)     
limitedTest$has_persref <- as.numeric(limitedTest$has_persref)     
limitedTest$cat_civstatus <- as.numeric(limitedTest$cat_civstatus)   
limitedTest$cat_residtype <- as.numeric(limitedTest$cat_residtype)    
limitedTest$num_account <- as.numeric(limitedTest$num_account)    
limitedTest$num_samejob <- as.numeric(limitedTest$num_samejob)       
limitedTest$num_sameplace_1 <- as.numeric(limitedTest$num_sameplace_1) 

preproc <- preProcess(limitedTrain)
normTrain <- predict(preproc, limitedTrain)
normTest <- predict(preproc, limitedTest)

set.seed(825)
Kmeans <- kmeans(normTrain, centers = 3)
Cluster <- Kmeans$cluster
table(Cluster)

km.kcca <- as.kcca(Kmeans, normTrain)
clusterTrain <- predict(km.kcca)
clusterTest <- predict(km.kcca, newdata=normTest)
table(clusterTrain)
table(clusterTest)

# apply clustering to youngsters
train1 <- subset(train.youngsters, clusterTrain == 1)
train2 <- subset(train.youngsters, clusterTrain == 2)
train3 <- subset(train.youngsters, clusterTrain == 3)

test1 <- subset(test.youngsters, clusterTest == 1)
test2 <- subset(test.youngsters, clusterTest == 2)
test3 <- subset(test.youngsters, clusterTest == 3)

# remove AgeGroup from the clustered training sets
train1 <- select(train2, -AgeGroup)
train2 <- select(train2, -AgeGroup)
train3 <- select(train3, -AgeGroup)
train4 <- select(train4, -AgeGroup)
train5 <- select(train5, -AgeGroup)

# store customer id vectors for each of the test set's clusters
customer_id1 <- test1$customer_id
customer_id2 <- test2$customer_id
customer_id3 <- test3$customer_id
customer_id4 <- test4$customer_id
customer_id5 <- test5$customer_id

#remove customer_id and AgeGroup from clustered data sets
test1 <- select(test1, -customer_id, -AgeGroup)
test2 <- select(test2, -customer_id, -AgeGroup)
test3 <- select(test3, -customer_id, -AgeGroup)
test4 <- select(test4, -customer_id, -AgeGroup)
test5 <- select(test5, -customer_id, -AgeGroup)

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
p5_1 <- svmLinearModeler (train5, test5, formula, cost_factor=0.1) #,gamma=1,coef0=0.004)

# predict with RF
set.seed(825)
p1_2 <- rfModeler (train1, test1, formula, alpha=1501)
p2_2 <- rfModeler (train2, test2, formula, alpha=1501)
p3_2 <- rfModeler (train3, test3, formula, alpha=1501) 
p4_2 <- rfModeler (train4, test4, formula, alpha=1501) 
p5_2 <- rfModeler (train5, test5, formula, alpha=1501) 

# predict with xgboost
set.seed(825)
p1_3 <- xgboostModeler (train1, test1, formula, mfinal=4, maxdepth=2)
p2_3 <- xgboostModeler (train2, test2, formula, mfinal=4, maxdepth=2)
p3_3 <- xgboostModeler (train3, test3, formula, mfinal=4, maxdepth=2)
p4_3 <- xgboostModeler (train4, test4, formula, mfinal=4, maxdepth=2)
p5_3 <- xgboostModeler (train5, test5, formula, mfinal=4, maxdepth=2)

# predict with nnet
set.seed(825)
p1_4 <- nnetModeler (train1, test1, formula, mfinal=50)
p2_4 <- nnetModeler (train2, test2, formula, mfinal=50)
p3_4 <- nnetModeler (train3, test3, formula, mfinal=50)
p4_4 <- nnetModeler (train4, test4, formula, mfinal=50)
p5_4 <- nnetModeler (train5, test5, formula, mfinal=50)

# predict with adaboost implementation from ada
set.seed(825)
p1_5 <- adaboostModeler (train1, test1, formula, mfinal=35, maxdepth=5, minsplit = 15)
p2_5 <- adaboostModeler (train2, test2, formula, mfinal=35, maxdepth=5, minsplit = 15)
p3_5 <- adaboostModeler (train3, test3, formula, mfinal=35, maxdepth=5, minsplit = 15)
p4_5 <- adaboostModeler (train4, test4, formula, mfinal=35, maxdepth=5, minsplit = 15)
p5_5 <- adaboostModeler (train5, test5, formula, mfinal=35, maxdepth=5, minsplit = 15)

n.models <-5

prediction1 <- (p1_1 + p1_2 + p1_3 + p1_4 + p1_5)/n.models 
prediction2 <- (p2_1 + p2_2 + p2_3 + p2_4 + p2_5)/n.models 
prediction3 <- (p3_1 + p3_2 + p3_3 + p3_4 + p3_5)/n.models 
prediction4 <- (p4_1 + p4_2 + p4_3 + p4_4 + p4_5)/n.models 
prediction5 <- (p5_1 + p5_2 + p5_3 + p5_4 + p5_5)/n.models 

# Output submission
threshold <- 0.5
print(paste("Output",Sys.time()))
PredTestLabels1 <- as.factor(ifelse(prediction1 > threshold, "1.0", "0.0"))
PredTestLabels2 <- as.factor(ifelse(prediction2 > threshold, "1.0", "0.0"))
PredTestLabels3 <- as.factor(ifelse(prediction3 > threshold, "1.0", "0.0"))
PredTestLabels4 <- as.factor(ifelse(prediction4 > threshold, "1.0", "0.0"))
PredTestLabels5 <- as.factor(ifelse(prediction5 > threshold, "1.0", "0.0"))
df1 <- data.frame(customer_id = customer_id1, target = PredTestLabels1)
df2 <- data.frame(customer_id = customer_id2, target = PredTestLabels2)
df3 <- data.frame(customer_id = customer_id3, target = PredTestLabels3)
df4 <- data.frame(customer_id = customer_id4, target = PredTestLabels4)
df5 <- data.frame(customer_id = customer_id5, target = PredTestLabels5)
MySubmission <- rbind(df1,df2,df3,df4,df5)
write.csv(MySubmission, output.csv.file, row.names=FALSE)

print(paste("Elapsed time: ",Sys.time()-strt))

# That's all!
########################################

