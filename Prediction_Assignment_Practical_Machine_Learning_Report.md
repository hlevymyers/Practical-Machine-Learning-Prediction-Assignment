# Prediction Assignment Practical Machine Learning
Helen Levy-Myers  
April 4, 2018  



## Background
This report examines the effort taken to find a prediction model for dumbbell bicep curls. The data set*  was designed to examine how "well" six individuals did bicep curls. Each person did biceps curls correctly, then made four different errors, throwing their elbows to the front, lifting only halfway, lowering only halfway, and throwing their hips to the front. Sensors added to the dumbbell, the waist belt, forearm and upper arm recorded data in x, y, and z axes plus other variables include roll, pitch, yaw, gryos, acceleration, time, user name, kurtosis, skewness, amplitude, magnet, with calculated averages, variation, standard deviation, for a total of 160 variables including the variable that identifies how the curl was performed, correctly or which error. 

If one does not regularly do bicep curls, one might not understand how they are done properly or what mistakes might look like. YouTube fortunately has many videos doing "Unilateral dumbbell biceps curls" and several more on "Top Mistakes Doing Bicep Curls." After watching numerous videos, one will understand that the upper arm and waist belt barely move when doing bicep curls properly. When moving the elbows to the front incorrectly, the dumbbell has a higher arc. Throwing the hips to the front should be indicated on the belt sensor and not necessarily during other movements.

## Cleaning Data
The data set is quite large with more than 19,000 observations and 160 variables. A very small testing data set was provided for final analysis. As best practice a validation test was created of 30 percent of the data and an exploring data set of 2,000 observations. The research was done on the exploring data until the final model was decided upon. The original data set included many columns with NAs and blank cells as they were columns with calculations across many observations. These columns were eliminated from the exploring data set and reduced the data set to 53 columns. The outcome variable was the "classe" of bicep curl and was originally a character variable. As part of the cleaning process it was converted to a factor variable with five levels, A, B, C, D, and E.

## Exploratory Analysis
After watching many videos, one could guess that a belt movement variable would be important. Some quick plots were done to look at the belt movement variable data. There are 38 variables measuring some aspect of belt movement. The E classe variable identified the throwing the hips forward movement which was easy to find in some plots. 

```r
#create validation test set
inTrain <- createDataPartition(y=dumbbell$classe, p =.7, list = FALSE)
training <- dumbbell[inTrain, ]
validation <- dumbbell[-inTrain, ]
training$classe <- as.factor(training$classe)
set.seed(123)
exploring <- sample_n(training, 2000)
exploring <- as.data.frame(exploring)

#data set without na values
exploringy <- select(exploring, 1, 9:12, 38:50, 61:69, 85:87, 103, 114:125, 141, 152:160) 
```

```r
#plot that separates throwing hips to front well, lowering halfway somewhat, other movements not separated
idea2 <- ggplot(exploring, aes(x = (roll_forearm+magnet_dumbbell_y), y = roll_belt, colour = factor(classe))) 
idea2 + geom_point(size = 2, alpha =.3)
```

![](Prediction_Assignment_Practical_Machine_Learning_Report_files/figure-html/unnamed-chunk-2-1.png)<!-- -->

## Models
Being able to predict one classe was possible, but the goal was to predict all five different classes. Using the caret package, several different models were examined including the rf, rpart, rpart2, ada, and pca methods. The rpart2 method was able to build a tree that had all five classes in separate leaves. The seven variables were used to build other models. Various models were tried. Although it could separate the classes, the accuracy was at best 90% and a higher accuracy was desired. 


```r
idea6 <- train(classe~., method ="rpart2", data = exploringy) #finds seven variables inlcudes tuning parameters
print(idea6$finalModel)
```

```
## n= 2000 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##  1) root 2000 1425 A (0.29 0.2 0.17 0.17 0.17)  
##    2) roll_belt< 130.5 1832 1260 A (0.31 0.22 0.19 0.19 0.095)  
##      4) pitch_forearm< -34.2 169    0 A (1 0 0 0 0) *
##      5) pitch_forearm>=-34.2 1663 1260 A (0.24 0.24 0.21 0.21 0.1)  
##       10) roll_forearm< 126.5 1070  704 A (0.34 0.27 0.13 0.2 0.055)  
##         20) magnet_dumbbell_y< 426.5 868  510 A (0.41 0.21 0.15 0.18 0.044) *
##         21) magnet_dumbbell_y>=426.5 202   95 B (0.04 0.53 0.03 0.3 0.1)  
##           42) total_accel_dumbbell>=5.5 127   29 B (0.063 0.77 0.047 0.0079 0.11) *
##           43) total_accel_dumbbell< 5.5 75   16 D (0 0.12 0 0.79 0.093) *
##       11) roll_forearm>=126.5 593  390 C (0.062 0.19 0.34 0.21 0.19)  
##         22) magnet_dumbbell_y< 291.5 318  143 C (0.082 0.1 0.55 0.14 0.13)  
##           44) magnet_forearm_z< -238 28    8 A (0.71 0.11 0 0.036 0.14) *
##           45) magnet_forearm_z>=-238 290  115 C (0.021 0.1 0.6 0.14 0.13) *
##         23) magnet_dumbbell_y>=291.5 275  191 D (0.04 0.28 0.1 0.31 0.27)  
##           46) accel_forearm_x>=-55 156   94 E (0.038 0.33 0.15 0.083 0.4) *
##           47) accel_forearm_x< -55 119   48 D (0.042 0.23 0.034 0.6 0.1) *
##    3) roll_belt>=130.5 168    3 E (0.018 0 0 0 0.98) *
```

```r
#final leaves of tree with seven variable and probabilities of each classification (A, B, C, D, E)
```
To get better accuracy, another direction was taken to add all the variables in using the dot option and change the method. Three classification methods were chosen, rf (random forest) which had an out of bounds error rate of 5.15% or accuracy rate of 94.85%, svmRadical, which had a significantly higher out of bounds error rate, and xgboost. 

## Best Fit
The xgboost method, or extreme gradient boosting, was determined to be the best model. It has almost perfect prediction across the entire exploring data set. The only problem with this model is that it takes a long time to run for large data sets as is used in this report. The Accuracy rate with the validation data set of 30% of the training set is 96.04%. As with the plot, classe E, throwing the hips forward, is the easiest to separate and has the lowest misapplied rate with less than 10 misapplied observations out of more than 1,000 E classe. 

```r
validation$classe <- as.factor(validation$classe)
validationy <- select(validation, 1, 9:12, 38:50, 61:69, 85:87, 103, 114:125, 141, 152:160)
idea11 <- train(classe~ ., method= "xgbTree", data = exploringy) #The BEST model
ideaPred11 <- predict(idea11, newdata = validationy)
confusionMatrix(ideaPred11, validationy$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1644   31    0    2    8
##          B   20 1077   52    0   10
##          C    9   25  949   21    8
##          D    0    0   22  940   21
##          E    1    6    3    1 1035
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9592          
##                  95% CI : (0.9538, 0.9641)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9484          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9821   0.9456   0.9250   0.9751   0.9566
## Specificity            0.9903   0.9827   0.9870   0.9913   0.9977
## Pos Pred Value         0.9757   0.9292   0.9377   0.9563   0.9895
## Neg Pred Value         0.9929   0.9869   0.9842   0.9951   0.9903
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2794   0.1830   0.1613   0.1597   0.1759
## Detection Prevalence   0.2863   0.1969   0.1720   0.1670   0.1777
## Balanced Accuracy      0.9862   0.9641   0.9560   0.9832   0.9771
```

## Cross Validation
To cross validate the model, a validation data set was created at beginning with 30% of the training set. The accuracy measure for the validation data set was 95.63%. Finally, the entire training set was run and its accuracy measure was 96.71%. 

```r
training <- as.data.frame(training)
trainingy <- select(training, 1, 9:12, 38:50, 61:69, 85:87, 103, 114:125, 141, 152:160) #data set without na values
ideaPred11a <- predict(idea11, newdata = trainingy)
confusionMatrix(ideaPred11a, trainingy$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3860   67    0    0   10
##          B   36 2540   77    0   23
##          C    9   41 2250   73   17
##          D    0    2   63 2173   32
##          E    1    8    6    6 2443
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9657          
##                  95% CI : (0.9625, 0.9687)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9566          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9882   0.9556   0.9391   0.9649   0.9675
## Specificity            0.9922   0.9877   0.9877   0.9916   0.9981
## Pos Pred Value         0.9804   0.9492   0.9414   0.9573   0.9915
## Neg Pred Value         0.9953   0.9893   0.9871   0.9931   0.9927
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2810   0.1849   0.1638   0.1582   0.1778
## Detection Prevalence   0.2866   0.1948   0.1740   0.1652   0.1794
## Balanced Accuracy      0.9902   0.9717   0.9634   0.9782   0.9828
```

## Conclusion
The best model to classify bicep curl movements is made using the extreme gradient boosting method in the caret package with all the variables. This provides an error rate less than 5% for the exploratory, validation and training data sets. 


*Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.
