# Overview
This repo represents a notebook with various cross-validation, prediction, and submission scripts toward the problem to predict the credit risk of potential loan applicants in one of the Brazilian banks. 

This is a binary classification problem that was the subject to the respective Kaggle competition (https://inclass.kaggle.com/c/adatelemz-si-platformok-2016-2-gyakorlat/).

The sections below briefly discuss the approach used to attack this prediction challenge.

# Data Exploration and Feature Engineering
First of all, it shall be mentioned the training data set was unbalanced in terms of observance ratio of the prediction class labels (target, the dependent variable). The label for no credit risk (“0.0”) was present in almost 73.5% of records whereas the target risk label (“1.0”) was observed for roughly 26.5% cases there, respectively. So this problem had a flavor of anomaly detection in it.
In terms of independent variables (predictors), there had been a subset of raw data features to disqualify from the model training. The variables disqualified in the pre-processing phase were as follows
- Noisy data without real business value: cat_dem, cat_zip1, cat_zip2, cat_zip3, cat_zip4, , customer_id
- Predictors with huge amount of NA values: cat_pinconf
- Variables introduced notable prediction accuracy degrading variance: cat_bsresid, num_sameplace, num_ndep, cat_opstate, cat_occupid, year, month

On top of that, there were variables for annual net income (num_income) and age of a credit applicant (num_age) that had great potential for the modelling yet displayed significant value variation. For them, the new aggregator features (IncomeGroup and AgeGroup, respectively) have been constructed. The original raw features have been then eliminated from the data frames used for model training and prediction.

*Note:* unfortunately the data code book provided for the training and data set was not explanatory enough in every piece. Although the majority of feature variables had self-descriptive names, two variables (year, month) were not well comprehensive. Therefore there was no any rationale to try to use those data in the instrumental feature engineering. In turn, keeping them in either raw or log-transformed form in the data set for model training and prediction degraded accuracy. Therefore they were simply excluded from the subset of qualified predictors.

# Data imputation
Since only a tiny fraction of records had NA in 5 features within the subset of qualified predictors, it had been decided to simply fill those NA with the most frequent value for each of those variables within the data set. The most frequent values were selected based on the aggregated statistics for a compound data frame where observations from original training and testing sets were combined. As a rule of thumb, data imputation on the aggregate of training and testing data gains better edge.

# Over-sampling
Since the original training data was severely unbalanced on the predictable class labels (see above), the straight-forward use of classification algorithms would lead to biased prediction toward the most frequent class label. This would therefore result in low accuracy of predictions on out-of-sample data.
The common recommendation to address such a problem is to do relevant over-sampling to increase the ratio of observations with the rare class label. The famous SMOTE technique has been applied for such an over-sampling.
Since the right methodology of assessing the outcome of SMOTE is to make it the part of cross-validation (rather than pre-processing of raw data), it required to develop custom cross-validation routines. Those routines ensured SMOTE transformation was plugged into the standard N-fold cross-validation flow.

# Cluster-Then-Predict
The concept of “cluster-then-predict” is a well-known technique to improve classification accuracy. In this case, it was also fruitful. The following clustering approaches have been tried
1. K-Mean clustering (with 5 clusters constructed)
2. Smart clustering based on business drivers
  a.Clustering by Age Group (with 4 clusters constructed)
  b.Clustering by Income Group (with 5 clusters constructed)
3. Mixed approach with clustering by Age Groups and then doing additional K-means 3-class clustering within younger people subpopulation

The best prediction results have been achieved with clustering by Age Groups. Such an approach was used when preparing the final submission for the competition.

# Model Training and Cross-Validation
In course of tackling the challenge, the following models have been trained to do binary classification
1. Xgboost (via caret interface)
2. GBM (via caret interface)
3. SVM with linear kernel (btw, this is one of the standard algorithms of choice of professional credit risk analysts engaged with real-world investment and retail banking)
4. Random Forests (using the implementation in randomForest library)
5. Adaboost classification (using capabilities of adabag library)
6. Old-generation neural netwok algorithm (using nnet library)

Predictions from those models were then used in ensembles to achieve better prediction accuracy.
Xgboost demonstrated the best individual performance on 5-fold cross-validation. The second best runners (with almost equal individual prediction accuracy) were SVM and GBM, with RF being a little worse than those two models. Adaboost and nnet were less accurate yet still not bad to play in the ensembles.
Several other models (SVM with radial kernel, CART, logistic regression) were also tried. However, they were disqualified in cross-validation trials.

# Orchestration
R was used as a programming language and environment to implement all of the routines needed. The overall setup of the data science experiment toward the submission for this problem was as follows
1.	Read training and testing data set into R
2.	Do pre-processing and feature engineering as explained in section “Data Exploration and Feature Engineering” above
3.	Do data imputation
4.	Do SMOTE over-sampling of training data
5.	Generate clusters and split both the trainingand testing set over the clusters
6.	Apply trained models to each of the clustered subsets
7.	Generate prediction per each cluster subset and model pair
8.	Combine predictions from various models into an ensemble
9.	Combine ensemble predictions for each cluster into a single prediction set
10.	Generate a submission file for Kaggle
Cross-validation has been performed in a separate loop, using the set of specific cross-validation scripts developed for every type of modelling algorithms tried. If cross-validation displayed good enough individual accuracy for a particular model, it was qualified to enter a submission ensemble.
The final submission was generated via weighted ensemble technique. In this setup, predictions from better performing models (xgboost, svm linear, and RF) got higher weights, and less-performing models (adaboost, nnet) got lower weights.

# Statistics of prediction accuracy
Below is track of classification accuracy of different submissions completed in the course of the competition
1. Data pre-processing, feature engineering and Straight-forward application of the modeling algorithms scored at 0.51-0.525 on the public leader board
2. Introducing “cluster-then-predict” technique raised prediction scores to 0.52-0.532 along
3. Adding SMOTE transformation further raised performance of individual prediction models to 0.572-0.602
4. Various ensembles with equal weights of different model predictions boosted the submission score to 0.615-0.635

# Files in this Notebook
- Example of equal-weight ensemble submission: loan_repay_ensemble_clustered_submission6.r
- Example of weighted ensemble submission within the clustering by Age groups setup: loan_repay_ensemble_clustered_submission8.r
- Example of weighted ensemble submission within the combined clustering by Age groups and K-mean 3-class clustering of younger people population
- Example of sole prediction model submission: loan_repay_xgboost_submission5.r
- Example of the custom cross-validation tool: loan_repay_rf_clustered_cv.r

The final ensemble with weighted predictions yielded the score of 0.644

