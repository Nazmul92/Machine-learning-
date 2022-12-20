## Machine learning based credit card fraud detection using data resampling.
In  this project, I focused on different resampling techniques to balance the imbalance data. After that I evaluated different machine learning algorithms to deect fradualant transaction.
## Problem definition
It is difficult to design machine learning (ML)-based fraudulent credit card detection models. Due to the extremely imbalanced class distributions and to improve accuracy, models analyze the data and always predict the majority class.
## Objectives
For balancing the data, I have concentrated on several data resampling techniques like random over-sampling, and random under-sampling, SMOTE (Synthetic Minority Oversampling TEchnique), and ROSE (Random Over-Sampling Examples).
## Dataset
In this project 
[creditcard.csv](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
is used.

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
```
