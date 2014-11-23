---
title: "Practical Machine Learning Project - Human Activity Recognition"
author: "Falko Schönteich"
date: "Sunday, November 23, 2014"
ActualVsPrediction_Frame: html_document
---

# Synopsis
In this study three prediction models (tree based, bagging based and random forest based) for human activity recognition (HAR) are created. It analysis data collected from six male participants (aged between 20-28) doing unilateral dumbbell Biceps curl. The data is categorized into 5 classes (A: exactly accoding to the specification; B:throwing the elbows to the front; C: lifting the dumbbell only halfway; D: lowering the dumbbell only halfway; E: throwing the hips to the front)

Data source: Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.


# Data Retrieval

The data can be retrieved from the following URLs:


```r
trainurl<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testurl<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
```

```r
trainfile<-"pml-training.csv"
testfile<-"pml-testing.csv"

# Download files if not already done.
#download.file(trainurl,trainfile)
#download.file(testurl,testfile)
```


```r
data_train<-read.csv(trainfile,na.string=c("NA",""))
data_test<-read.csv(testfile,na.string=c("NA",""))
```


# Data Preprocessing
Dividing the data in a training and a testing set.

```r
library(caret)
set.seed(123)
inTrain = createDataPartition(y=data_train$classe, p=0.7, list=FALSE)
training = data_train[inTrain,]
testing = data_train[-inTrain,]

dim(training)
```

```
## [1] 13737   160
```

```r
table(training$classe)
```

```
## 
##    A    B    C    D    E 
## 3906 2658 2396 2252 2525
```

Some of the columns have almost no data, so they can be left out.

```r
na_count = sapply(training, function(x) {sum(is.na(x))})
table(na_count)
```

```
## na_count
##     0 13448 
##    60   100
```

```r
empty_columns= names(na_count[na_count>=13440])
training = training[, !names(training) %in% empty_columns]
names(training)
```

```
##  [1] "X"                    "user_name"            "raw_timestamp_part_1"
##  [4] "raw_timestamp_part_2" "cvtd_timestamp"       "new_window"          
##  [7] "num_window"           "roll_belt"            "pitch_belt"          
## [10] "yaw_belt"             "total_accel_belt"     "gyros_belt_x"        
## [13] "gyros_belt_y"         "gyros_belt_z"         "accel_belt_x"        
## [16] "accel_belt_y"         "accel_belt_z"         "magnet_belt_x"       
## [19] "magnet_belt_y"        "magnet_belt_z"        "roll_arm"            
## [22] "pitch_arm"            "yaw_arm"              "total_accel_arm"     
## [25] "gyros_arm_x"          "gyros_arm_y"          "gyros_arm_z"         
## [28] "accel_arm_x"          "accel_arm_y"          "accel_arm_z"         
## [31] "magnet_arm_x"         "magnet_arm_y"         "magnet_arm_z"        
## [34] "roll_dumbbell"        "pitch_dumbbell"       "yaw_dumbbell"        
## [37] "total_accel_dumbbell" "gyros_dumbbell_x"     "gyros_dumbbell_y"    
## [40] "gyros_dumbbell_z"     "accel_dumbbell_x"     "accel_dumbbell_y"    
## [43] "accel_dumbbell_z"     "magnet_dumbbell_x"    "magnet_dumbbell_y"   
## [46] "magnet_dumbbell_z"    "roll_forearm"         "pitch_forearm"       
## [49] "yaw_forearm"          "total_accel_forearm"  "gyros_forearm_x"     
## [52] "gyros_forearm_y"      "gyros_forearm_z"      "accel_forearm_x"     
## [55] "accel_forearm_y"      "accel_forearm_z"      "magnet_forearm_x"    
## [58] "magnet_forearm_y"     "magnet_forearm_z"     "classe"
```

The first 7 variables are only informational, so they can be ommitted as well.

```r
training = training[,-c(1:7)]
```

# Prediction models
Three prediction models are created with different methods (tree based, bagging based, random forest based).

### Tree based model

```r
library(rpart)
rpart_model <-train(classe ~ ., method="rpart",data = training)
```

### Bagging based model

```r
library(ipred)
bag_model <- bagging(classe ~ ., data = training, coob = T)
```

### Random forest based model

```r
library(randomForest)
rf_model <- randomForest(classe ~ ., data = training, importance = T)
```

#Evaluating the models
First, a data frame for data comparison is created.

```r
ActualVsPrediction_Frame<-data.frame(ActualClass = testing$classe, PredictionTree = predict(rpart_model, testing), PredictionBagging = predict(bag_model, testing), PredictionRandomForest = predict(rf_model, testing))
```

## Correct predictions
To determine the precision of the chosen models, the actual values are compared to the predictions.

```r
nrow(testing)
```

```
## [1] 5885
```

```r
sum(ActualVsPrediction_Frame$ActualClass==ActualVsPrediction_Frame$PredictionTree)
```

```
## [1] 3250
```

```r
sum(ActualVsPrediction_Frame$ActualClass==ActualVsPrediction_Frame$PredictionBagging)
```

```
## [1] 5810
```

```r
sum(ActualVsPrediction_Frame$ActualClass==ActualVsPrediction_Frame$PredictionRandomForest)
```

```
## [1] 5854
```

## Confusion marix
To further analyse the accuracy of the prediction models, the confusion matrices are calculated.

```r
confusionMatrix(ActualVsPrediction_Frame$PredictionTree,ActualVsPrediction_Frame$ActualClass)
```

```
## Loading required namespace: e1071
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1061  235   27   64   13
##          B  163  631   42  133  281
##          C  341  230  819  509  247
##          D  102   43  138  258   60
##          E    7    0    0    0  481
## 
## Overall Statistics
##                                         
##                Accuracy : 0.552         
##                  95% CI : (0.539, 0.565)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.437         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.634    0.554    0.798   0.2676   0.4445
## Specificity             0.919    0.870    0.727   0.9303   0.9985
## Pos Pred Value          0.758    0.505    0.382   0.4293   0.9857
## Neg Pred Value          0.863    0.890    0.945   0.8664   0.8886
## Prevalence              0.284    0.194    0.174   0.1638   0.1839
## Detection Rate          0.180    0.107    0.139   0.0438   0.0817
## Detection Prevalence    0.238    0.212    0.365   0.1021   0.0829
## Balanced Accuracy       0.777    0.712    0.763   0.5990   0.7215
```

```r
confusionMatrix(ActualVsPrediction_Frame$PredictionBagging,ActualVsPrediction_Frame$ActualClass)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1668   11    2    0    0
##          B    5 1116   10    0    0
##          C    0    9 1010   26    2
##          D    0    3    4  937    1
##          E    1    0    0    1 1079
## 
## Overall Statistics
##                                        
##                Accuracy : 0.987        
##                  95% CI : (0.984, 0.99)
##     No Information Rate : 0.284        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.984        
##  Mcnemar's Test P-Value : NA           
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.996    0.980    0.984    0.972    0.997
## Specificity             0.997    0.997    0.992    0.998    1.000
## Pos Pred Value          0.992    0.987    0.965    0.992    0.998
## Neg Pred Value          0.999    0.995    0.997    0.995    0.999
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.283    0.190    0.172    0.159    0.183
## Detection Prevalence    0.286    0.192    0.178    0.161    0.184
## Balanced Accuracy       0.997    0.988    0.988    0.985    0.998
```

```r
confusionMatrix(ActualVsPrediction_Frame$PredictionRandomForest,ActualVsPrediction_Frame$ActualClass)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    4    0    0    0
##          B    1 1135   11    0    0
##          C    0    0 1015   14    0
##          D    0    0    0  949    0
##          E    0    0    0    1 1082
## 
## Overall Statistics
##                                         
##                Accuracy : 0.995         
##                  95% CI : (0.993, 0.996)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.993         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.999    0.996    0.989    0.984    1.000
## Specificity             0.999    0.997    0.997    1.000    1.000
## Pos Pred Value          0.998    0.990    0.986    1.000    0.999
## Neg Pred Value          1.000    0.999    0.998    0.997    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.172    0.161    0.184
## Detection Prevalence    0.285    0.195    0.175    0.161    0.184
## Balanced Accuracy       0.999    0.997    0.993    0.992    1.000
```

# Conclusion
As the confusion matrix for the random forest model shows the lowest error rate (Accuracy for tree based:0.553
; for bagging based: 0.9876; for random forest based: 0.9947), this model produces the best predictions.

# Predicting assignement set
With the random forest model, the assignment sets are predicted.

```r
answers <- predict(rf_model, data_test)

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(answers)

answers
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

All the predictions are correct.
