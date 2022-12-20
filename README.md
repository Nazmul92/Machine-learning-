## Machine learning based credit card fraud detection using data resampling.
In this project, I focused on different resampling techniques to balance the imbalanced data. After that, I evaluated different machine learning algorithms to detect fraudulent transactions.
## Problem definition
It is difficult to design machine learning (ML)-based fraudulent credit card detection models. Due to the extremely imbalanced class distributions and to improve accuracy, models analyze the data and always predict the majority class.
## Objectives
For balancing the data, I have concentrated on several data resampling techniques like random over-sampling, and random under-sampling, SMOTE (Synthetic Minority Oversampling TEchnique), and ROSE (Random Over-Sampling Examples).
## Dataset
In this project 
[creditcard.csv](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
 dataset is used.

## Exploratory data analysis (EDA)

Load dataset

```bash
data<-read.csv(‘creditcard.csv’)
```
Check dataset dimension
```bash
dim(data)
284807     31
```
Check missing value
```bash
colSums(is.na(df))
Time 0:V1 0: V2 0: V3 0: V4 0: V5 0: V6 0: V7 0: V8 0: V9 0: V10 0: V11 0: V12 0: V13 0: V14 0: V15 0: V16 0: V17 0: V18 0: V19 0: V20 0: V21 0: V22 0: V23 0: V24 0: V25 0: V26 0: V27 0: V28 0: Amount 0: Class 0

```
Check class imbalance
```bash
table(data$Class)
     0      1 
284315    492
```
Make data subset (10%) of whole data.
```bash
library(dplyr)
creditcard<-data %>% sample_frac(0.1)
```
Check the dimension of the subset
```bash
library(dplyr)
creditcard<-data %>% sample_frac(0.1)
```
## Data resampling techniques
Four data resampling techniques ( random under-sampling, random over-sampling, SMOTE, and ROSE is employed in this project.
## Random under sampling
>> Random undersampling randomly selects and removes samples from the majority class.
>> This method discards enormous amounts of data, which is quite troublesome because it can make it more difficult to learn the decision border between the minority and majority samples, which can lead to a reduction in classification performance.
```bash
library(caret)
down_train<-downSample(x=train_data, 
y = as.factor(train_data$Class))
table(down_train$Class)
 0  1 
42 42 
```
## Random over-sampling
>>Random oversampling selects a random sample from the minority class and adds multiple copies of this instance to the training data.
>>The random oversampling may increase the likelihood of overfitting occurring since it makes exact copies of the minority class samples.
```bash
library(caret)
up_train<-upSample(x=train_data,
y = as.factor(train_data$Class))
table(up_train$Class)
    0     1 
22743 22743 
```
## SMOTE (Synthetic Minority Oversampling Technique) 
>>SMOTE is a type of oversampling that works differently from normal oversampling.
It generates synthetic data using a k-nearest neighbor algorithm. 
It begins by selecting random data from the minority class, and then the k-nearest neighbors are determined for this data point.
It computes the vector between the current data point and the selectedneighbor using one of those neighbors.
A random number between 0 and 1 is multipliedwith the vector, and this is added to the current data point.
```bash
library(smotefamily)
smote_train<-SMOTE(train_data[,-30], train_data$Class, K=10)
smote_train<-smote_train$data
table(smote_train$class)
    0     1 
22745 22720 
```
## ROSE (Random Over-Sampling Examples) 
>>ROSE aids the task of binary classification in the presence of minority classes.
It produces a synthetic, possibly balanced, sample of data simulated according to a smoothed bootstrapping approach.
```bash
library(ROSE)
rose_train<-ROSE(Class~., train_data)
rose_train<-rose_train$data
table(rose_train$Class)
    0     1 
11466 11319
```
## ML algorithms evaluation
Four different ML algorithms (logistic regression, decision tree, k-nearest neighbor, and naive bayes) are employed in this project. Each ML algorithm is evaluated for imbalanced data and four resampled data based on sensitivity, specificity, recall, the f1-score, and the ROC curve.



