library(dplyr) 
library(caret) 
library(caTools)
library(ggplot2)
library(corrplot) 
library(rpart)
library(smotefamily)
library(ROSE)
library(pROC)
library(class)
library(naivebayes)

##load data
data<-read.csv("creditcard.csv")
dim(data)                                # check dimension
str(data)                                # check data structure
colSums(is.na(data))                     # check missing values
table(data$Class)                        # check class imbalance
prop.table(table(creditcard$Class))      # check class imbalance proportion
creditcard<-data %>% sample_frac(0.1)    # make subset (10%) of the dataset
creditcard<-creditcard[,-1]              # reduce the first column (time)


## bar plot of imbalanced data
p1<-ggplot(creditcard ,aes(x = factor(Class), y =prop.table(stat(count)), fill = factor(Class))) +
  geom_bar(position = "dodge") +
  scale_y_continuous(labels = scales::percent) +
  scale_x_discrete(labels = c("no fraud (0)", "fraud (1)"))+
  labs(x = 'Class', y = 'Percentage') +
  ggtitle("Imbalanced data")+
  theme(plot.title = element_text(hjust = 0.5))
 

## spliting data into training and testing set
data_sample<-sample.split(creditcard$Class, SplitRatio = 0.8)
train_data<-subset(creditcard, data_sample == TRUE)
test_data<-subset(creditcard, data_sample == FALSE)


creditcard$Class<-factor(creditcard$Class, levels = c(0,1))  # factor of class variable

## random under-sampling
set.seed(1)
down_train<-downSample(x=train_data, y = as.factor(train_data$Class))
table(down_train$Class)         # class distribution after under sampling


## barplot of class distribution following under sampling
p2<-ggplot(down_train[,-32] ,aes(x = factor(Class), y = prop.table(stat(count)), fill = factor(Class))) +
  geom_bar(position = "dodge") + 
  scale_y_continuous(labels = scales::percent) +
  scale_x_discrete(labels = c("no fraud (0)", "fraud (1)"))+
  labs(x = 'Class', y = 'Percentage') +
  ggtitle("Down sampling")+
  theme(plot.title = element_text(hjust = 0.5))
                                      
## random over-sampling
set.seed(2)
up_train<-upSample(x = train_data, y = as.factor(train_data$Class))
table(up_train$Class)              # class distribution following over sampling


## barplot of class distribution following over sampling
p3<-ggplot(up_train[,-32] ,aes(x = factor(Class), y = prop.table(stat(count)), fill = factor(Class))) +
  geom_bar(position = "dodge") +
  scale_y_continuous(labels = scales::percent) +
  scale_x_discrete(labels = c("Non fraud (0)", "Fraud (1)"))+
  labs(x = 'Class', y = 'Percentage') +
  ggtitle("Up sampling")+
  theme(plot.title = element_text(hjust = 0.5))  

##smote sampling
set.seed(1)
smote_train<-SMOTE(train_data[,-30], train_data$Class, K=10)
smote_train<-smote_train$data
str(smote_train)  
dim(smote_train)
smote_train[,-31]$Class<-as.factor(smote_train$Class)
table(smote_train$Class)
    
p4<-ggplot(smote_train[,-32] ,aes(x = factor(Class), y = prop.table(stat(count)), fill = factor(Class))) +
  geom_bar(position = "dodge") +
  scale_y_continuous(labels = scales::percent) +
  scale_x_discrete(labels = c("Non fraud (0)", "Fraud (1)"))+
  labs(x = 'Class', y = 'Percentage') +
  ggtitle("SMOTE sampling")+
  theme(plot.title = element_text(hjust = 0.5)) 
  
##rose sampling
  set.seed(1)
  rose_train<-ROSE(Class~., train_data)
  rose_train<-rose_train$data
  table(rose_train$Class)                # class distriution after smote sampling
  rose_train$Class<-as.factor(rose_train$Class)
  
## barplot of class distribution after smote sampling
  p5<-ggplot(rose_train[,-32] ,aes(x = factor(Class), y = prop.table(stat(count)), fill = factor(Class))) +
    geom_bar(position = "dodge") +
    scale_y_continuous(labels = scales::percent) +
    scale_x_discrete(labels = c("Non fraud (0)", "Fraud (1)"))+
    labs(x = 'Class', y = 'Percentage') +
    ggtitle("ROSE sampling")+
    theme(plot.title = element_text(hjust = 0.5)) 
  

## distribution of time variable by class
p1<-ggplot(data=creditcard,aes(x = Time, fill = factor(Class))) + geom_histogram(bins = 50,color='white')+
  labs(x = 'Time (s)', y = 'Number of transactions') +
  ggtitle('Distribution of transaction by class with time') +
  theme(plot.title = element_text(hjust = 0.5))+
  facet_grid(Class ~ ., scales = 'free_y')

## distribution of amount by class
ggplot(data=creditcard, aes(x= factor(Class), y= Amount))+
  geom_boxplot(outlier.color = "darkorange", outlier.size = 3)+
  scale_x_discrete(labels = c("Non fraud (0)", "Fraud (1)"))+
  ggtitle("Distribution of transaction amount by class")+
  theme(plot.title = element_text(hjust = 0.5))

## correlation among features
correlations <- cor(creditcard[,-1],method="pearson")
corrplot(correlations, number.cex = .9, method = "square",
         type = "full", tl.cex=0.5,tl.col = "darkblue")

##logistic regression for imbalanced data
set.seed(1)
log_mod_imb <- glm(Class ~ ., family = "binomial", data = train_data)
log_pred_imb<-predict(log_mod_imb, test_data, type = "response")
y_pred_prob<-ifelse(log_pred_imb>0.5,1,0)
y_pred_imb<-factor(y_pred_prob, levels = c(0,1))


## confusion matrix for logistic regression (imbalanced data)
conf_mat <- confusionMatrix(as.factor(y_pred_imb),as.factor(test_data$Class))
conf_mat
tp<-5682
tn<-8
fn<-3
fp<-4
precision<-tp/(tp+fp)
recall<-tp/(tp+fn)
f1_score<-(2*precision*recall)/(precision+recall)
tnr<-tn/(tn+fp)
auc_imb<-auc(as.numeric(test_data$Class), as.numeric(y_pred_imb))

##logistic regression for under-sampling
set.seed(1)
log_mod_down <- glm(Class ~ ., family = "binomial", data = down_train)
log_pred_down<-predict(log_mod_down, test_data, type = "response")
y_pred_prob<-ifelse(log_pred_down>0.5,1,0)
y_pred_down<-factor(y_pred_prob, levels = c(0,1))

##confusion matrix logistic regression (under-sampling)
conf_mat <- confusionMatrix(as.factor(y_pred_down),as.factor(test_data$Class))
conf_mat
tp<-5080
tn<-9
fn<-2
fp<-606
precision<-tp/(tp+fp)
recall<-tp/(tp+fn)
f1_score<-(2*precision*recall)/(precision+recall)
tnr<-tn/(tn+fp)
auc_down<-auc(as.numeric(test_data$Class), as.numeric(y_pred_down))


##logistic regression for over-sampling
set.seed(1)
log_mod_up<- glm(Class ~ ., family = "binomial", data = up_train)
log_pred_up<-predict(log_mod_up, test_data, type = "response")
y_pred_prob<-ifelse(log_pred_up>0.5,1,0)
y_pred_up<-factor(y_pred_prob, levels = c(0,1))

##confusion matrix of logistic regression (up-sampling)
conf_mat <- confusionMatrix(as.factor(y_pred_up),as.factor(test_data$Class))
conf_mat
tp<-5652
tn<-10
fn<-1
fp<-34
precision<-tp/(tp+fp)
recall<-tp/(tp+fn)
f1_score<-(2*precision*recall)/(precision+recall)
tnr<-tn/(tn+fp)
auc_up<-auc(as.numeric(test_data$Class), as.numeric(y_pred_up))   # calculate auc values


##logistic regression for smote-sampling
set.seed(1)
log_mod_smote<- glm(as.factor(class)~., data = smote_train, family = "binomial" )
length((smote_train[,30]))
log_pred_smote<-predict(log_mod_smote, test_data, type = "response")
y_pred_prob<-ifelse(log_pred_smote>0.5,1,0)
y_pred_smote<-factor(y_pred_prob, levels = c(0,1))


##confusion matrix logistic regression smote-sampling
conf_mat <- confusionMatrix(as.factor(y_pred_smote),as.factor(test_data$Class))
conf_mat
tp<-5664
tn<-10
fn<-1
fp<-22
precision<-tp/(tp+fp)
recall<-tp/(tp+fn)
f1_score<-(2*precision*recall)/(precision+recall)
tnr<-tn/(tn+fn)

auc_smote<-auc(as.numeric(test_data$Class), as.numeric(y_pred_smote))   # calculate auc values


##logistic regression for rose-sampling
set.seed(1)
log_mod_rose<- glm(Class~., family = "binomial", data = rose_train)
log_pred_rose<-predict(log_mod_rose, test_data, type = "response")
y_pred_prob<-ifelse(log_pred_rose>0.5,1,0)
y_pred_rose<-factor(y_pred_prob, levels = c(0,1))

##confusion matrix logistic regression rose-sampling
set.seed(1)
conf_mat <- confusionMatrix(as.factor(y_pred_rose),as.factor(test_data$Class))
conf_mat
tp<-5626
tn<-10
fn<-1
fp<-60
precision<-tp/(tp+fp)
recall<-tp/(tp+fn)
f1_score<-(2*precision*recall)/(precision+recall)
tnr<-tn/(tn+fn)

auc_rose<-auc(as.numeric(test_data$Class), as.numeric(y_pred_rose))  # calculate auc values

##roc curve of logistic regression for all resamplings and imbalanced data
set.seed(1)
par(pty = "s")
roc(test_data$Class,log_pred_imb, col = "deepskyblue", plot = TRUE, main = "ROC-Logistic Regression", 
    xlab = "False positive rate", ylab = "True positive rate",
    legacy.axes = TRUE,lwd=3)
plot(roc(test_data$Class,log_pred_smote,
         legacy.axes = TRUE, ), add = TRUE, col ="brown1",lwd=3)
plot(roc(test_data$Class,log_pred_down,
         legacy.axes = TRUE),  add = TRUE, col = "chartreuse3",lwd=3)
plot(roc(test_data$Class,log_pred_up,
         legacy.axes = TRUE), add = TRUE, col = "darkorchid1",lwd=3)
plot(roc(test_data$Class,log_pred_rose,
         legacy.axes = TRUE), add = TRUE, col = "gold2",lwd=3)
legend(0.5,0.25, cex = 0.6, text.font = 1,legend=c("Imbalanced (AUC = 0.89)", "Down-sampling (AUC = 0.80)","Up-sampling (AUC = 0.87)",
                                                   "SMOTE (AUC = 0.88)","ROSE (AUC = 0.89)"),
       col=c("deepskyblue", "chartreuse3", "darkorchid1", "brown1", "gold2"), lwd=3)


##decision tree for imbalanced data
set.seed(1)
d_tree_imb<-rpart(Class~.,data = train_data, method = "class")
pred_tree_imb<-predict(d_tree_imb,test_data, type = "class")

##confusion matrix decision tree
confusionMatrix(as.factor(pred_tree_imb),as.factor(test_data$Class))
tp<-5681
tn<-8
fn<-3
fp<-5
precision<-tp/(tp+fp)
recall<-tp/(tp+fn)
f1_score<-(2*precision*recall)/(precision+recall)
tnr<-tn/(tn+fn)
auc_dtree_imb<-auc(as.numeric(test_data$Class), as.numeric(pred_tree_imb))

## decision tree for under-sampling
set.seed(1)
d_tree_down<-rpart(Class~.,data = down_train, method = "class")
pred_tree_down<-predict(d_tree_down,test_data, type = "class")

## confusion matrix of decision tree (down-sampling)
confusionMatrix(as.factor(pred_tree_down),as.factor(test_data$Class))
tp<-4900
tn<-9
fn<-2
fp<-786
precision<-tp/(tp+fp)
recall<-tp/(tp+fn)
f1_score<-(2*precision*recall)/(precision+recall)
tnr<-tn/(tn+fn)

auc_dtree_down<-auc(as.numeric(test_data$Class), as.numeric(pred_tree_down))

## decision tree for up-sampling
set.seed(112)
d_tree_up<-rpart(Class~.,data = up_train, method = "class")
pred_tree_up<-predict(d_tree_up,test_data, type = "class")

## confusion matrix of decision tree (up-sampling)
confusionMatrix(as.factor(pred_tree_up),as.factor(test_data$Class))
tp<-5576
tn<-11
fn<-0
fp<-110
precision<-tp/(tp+fp)
recall<-tp/(tp+fn)
f1_score<-(2*precision*recall)/(precision+recall)
tnr<-tn/(tn+fn)

auc_dtree_up<-auc(as.numeric(test_data$Class), as.numeric(pred_tree_up)) # calculate auc values

## decision tree smote-sampling
set.seed(1)
smote_train$Class<-as.factor(smote_train$Class)    # make factor of class variable
d_tree_smote<-rpart(class~.,data = smote_train, method = "class")
pred_tree_smote<-predict(d_tree_smote,test_data, type = "class")

## confusion matrix of decision tree (smote-sampling)
confusionMatrix(as.factor(pred_tree_smote),as.factor(test_data$Class))
tp<-5573
tn<-11
fn<-0
fp<-113
precision<-tp/(tp+fp)
recall<-tp/(tp+fn)
f1_score<-(2*precision*recall)/(precision+recall)
tnr<-tn/(tn+fn)

auc_dtree_smote<-auc(as.numeric(test_data$Class), as.numeric(pred_tree_smote)) # calculate auc values

##decision tree for rose-sampling
set.seed(123)
d_tree_rose<-rpart(Class~.,data = rose_train, method = "class")
pred_tree_rose<-predict(d_tree_rose,test_data, type = "class")

##confusion matrix of decision tree (smote-sampling)
confusionMatrix(as.factor(pred_tree_rose),as.factor(test_data$Class))
tp<-5598
tn<-11
fn<-0
fp<-88
precision<-tp/(tp+fp)
recall<-tp/(tp+fn)
f1_score<-(2*precision*recall)/(precision+recall)
tnr<-tn/(tn+fn)

auc_dtree_rose<-auc(as.numeric(test_data$Class), as.numeric(pred_tree_rose))

## ROC of decision tree for all resampling and imabalaned data
set.seed(1)
par(pty = "s")
roc(as.numeric(test_data$Class),as.numeric(pred_tree_imb), col = "deepskyblue", plot = TRUE, main = "ROC: Decision Tree", 
    xlab = "False positive rate", ylab = "True positive rate",
    legacy.axes = TRUE,lwd = 3)
plot(roc(as.numeric(test_data$Class),as.numeric(pred_tree_smote),
         legacy.axes = TRUE, ), add = TRUE, col ="brown1", lwd = 3)
plot(roc(as.numeric(test_data$Class),as.numeric(pred_tree_down),
         legacy.axes = TRUE),  add = TRUE, col = "chartreuse3", lwd = 3)
plot(roc(as.numeric(test_data$Class),as.numeric(pred_tree_up),
         legacy.axes = TRUE), add = TRUE, col = "darkorchid1", lwd = 3)
plot(roc(as.numeric(test_data$Class),as.numeric(pred_tree_rose),
         legacy.axes = TRUE), add = TRUE, col = "gold2", lwd = 3)
legend(0.5,0.25, cex = 0.6, text.font = 1,legend=c("Imbalanced (AUC = 0.89)", "Down-sampling (AUC = 0.81)","Up-sampling (AUC = 0.88)",
                                                   "SMOTE (AUC = 0.88)","ROSE (AUC = 0.88)"),
       col=c("deepskyblue", "chartreuse3", "darkorchid1", "brown1", "gold2"), lwd=3)


##selecting optimal k for KNN
i=1                          
k.optm=1                     
for (i in 1:28){ 
  knn.mod <-  knn(train=train_data, test=test_data, cl=train_data$Class, k=i)
  k.optm[i] <- 100 * sum(test_data$Class == knn.mod)/NROW(test_data$Class)
  k=i  
  cat(k,'=',k.optm[i],'\n')       
}

## knn for imbalanced data
knn_classifier_imb<-knn(train = train_data, test = test_data, cl = train_data$Class,
                        k=10)

##confusion matrix KNN for imbalanced data 
confusionMatrix(as.factor(knn_classifier_imb),as.factor(test_data$Class))
tp<-5684
tn<-2
fn<-9
fp<-2
precision<-tp/(tp+fp)
recall<-tp/(tp+fn)
f1_score<-(2*precision*recall)/(precision+recall)
tnr<-tn/(tn+fn)

auc_knn_imb<-auc(as.numeric(test_data$Class), as.numeric(knn_classifier_imb))

## knn for down-sampling data
knn_classifier_down<-knn(train = down_train[,-32], test = test_data, cl = down_train$Class,
                         k=10)

## confusion matrix of KNN (down-sampling) 
confusionMatrix(as.factor(knn_classifier_down),as.factor(test_data$Class))
tp<-4662
tn<-3
fn<-8
fp<-1024
precision<-tp/(tp+fp)
recall<-tp/(tp+fn)
f1_score<-(2*precision*recall)/(precision+recall)
tnr<-tn/(tn+fn)

auc_knn_down<-auc(as.numeric(test_data$Class), as.numeric(knn_classifier_down))

##knn in up-sampling data
set.seed(112)
knn_classifier_up<-knn(train = up_train[,-32], test = test_data, cl = up_train$Class,
                       k=10)

##confusion matrix of KNN (up-sampling) 
confusionMatrix(as.factor(knn_classifier_up),as.factor(test_data$Class))
tp<-3054
tn<-5
fn<-6
fp<-2632
precision<-tp/(tp+fp)
recall<-tp/(tp+fn)
f1_score<-(2*precision*recall)/(precision+recall)
tnr<-tn/(tn+fn)

auc_knn_up<-auc(as.numeric(test_data$Class), as.numeric(knn_classifier_up))

##knn in smote-sampling 
set.seed(112)
knn_classifier_smote<-knn(train = smote_train, test = test_data, cl = smote_train[,30],
                          k=10)

##confusion matrix of KNN (smote-sampling) 
confusionMatrix(as.factor(knn_classifier_smote),as.factor(test_data$Class))
tp<-5616
tn<-10
fn<-1
fp<-70
precision<-tp/(tp+fp)
recall<-tp/(tp+fn)
f1_score<-(2*precision*recall)/(precision+recall)
tnr<-tn/(tn+fn)

auc_knn_smote<-auc(as.numeric(test_data$Class), as.numeric(knn_classifier_smote))

##knn for rose-sampling data
set.seed(115)
knn_classifier_rose<-knn(train = rose_train, test = test_data, cl = rose_train[,31],
                         k=10)

##confusion matrix of KNN rose-sampling 
confusionMatrix(as.factor(knn_classifier_rose),as.factor(test_data$Class))
tp<-5646
tn<-7
fn<-4
fp<-40
precision<-tp/(tp+fp)
recall<-tp/(tp+fn)
f1_score<-(2*precision*recall)/(precision+recall)
tnr<-tn/(tn+fn)

auc_knn_rose<-auc(as.numeric(test_data$Class), as.numeric(knn_classifier_rose))

## ROC of KNN for all samplings and imbalanced
set.seed(1)
par(pty = "s")
roc(as.numeric(test_data$Class),as.numeric(knn_classifier_imb), col = "deepskyblue", plot = TRUE, main = "ROC: K-nearest neighbor", 
    xlab = "False positive rate", ylab = "True positive rate",
    legacy.axes = TRUE, lwd = 3)
plot(roc(as.numeric(test_data$Class),as.numeric(knn_classifier_smote),
         legacy.axes = TRUE, ), add = TRUE, col ="brown1", lwd = 3)
plot(roc(as.numeric(test_data$Class),as.numeric(knn_classifier_down),
         legacy.axes = TRUE),  add = TRUE, col = "chartreuse3", lwd = 3)
plot(roc(as.numeric(test_data$Class),as.numeric(knn_classifier_up),
         legacy.axes = TRUE), add = TRUE, col = "darkorchid1", lwd = 3)
plot(roc(as.numeric(test_data$Class),as.numeric(knn_classifier_rose),
         legacy.axes = TRUE), add = TRUE, col = "cyan3", lwd = 3)
legend(0.5,0.25, cex = 0.6, text.font = 1,legend=c("Imbalanced (AUC = 0.50)", "Down-sampling (AUC = 0.62)","Up-sampling (AUC = 0.54)",
                                                   "SMOTE (AUC = 0.63)","ROSE (AUC = 0.50)"),
       col=c("deepskyblue", "chartreuse3", "darkorchid1", "brown1", "cyan3"), lwd=3)


## naive bayes for imbalanced data
train_data$Class<-as.factor(train_data$Class) 
set.seed(12)
model_bayes_imb<-naive_bayes(Class~.,data = train_data, usekernel = T)
pred_bayes_imb<-predict(model_bayes_imb,test_data)
             

## confusion matrix naive bayes imbalanced data
confusionMatrix(as.factor(pred_bayes_imb),as.factor(test_data$Class))
tp<-5625
tn<-10
fn<-1
fp<-61
precision<-tp/(tp+fp)
recall<-tp/(tp+fn)
f1_score<-(2*precision*recall)/(precision+recall)
tnr<-tn/(tn+fn)

auc_nbayes_imb<-auc(as.numeric(test_data$Class), as.numeric(pred_bayes_imb))

## naive bayes for under sampling
set.seed(1)
down_train$Class<-as.factor(down_train$Class)
model_bayes_down<-naive_bayes(Class~.,data = down_train, usekernel = T)
pred_bayes_down<-predict(model_bayes_down,test_data)

## confusion matrix of naive bayes (down-sampling data)
confusionMatrix(as.factor(pred_bayes_down),as.factor(test_data$Class))
tp<-5451
tn<-10
fn<-1
fp<-235
precision<-tp/(tp+fp)
recall<-tp/(tp+fn)
f1_score<-(2*precision*recall)/(precision+recall)
tnr<-tn/(tn+fn)

auc_nbayes_down<-auc(as.numeric(test_data$Class), as.numeric(pred_bayes_down))

##naive bayes for up-sampling
set.seed(101)
up_train$Class<-as.factor(up_train$Class)
model_bayes_up<-naive_bayes(Class~.,data = up_train, usekernel = T)
pred_bayes_up<-predict(model_bayes_up,test_data)

## confusion matrix of naive bayes up-sampling data
confusionMatrix(as.factor(pred_bayes_up),as.factor(test_data$Class))
tp<-5646
tn<-10
fn<-1
fp<-40
precision<-tp/(tp+fp)
recall<-tp/(tp+fn)
f1_score<-(2*precision*recall)/(precision+recall)
tnr<-tn/(tn+fn)

auc_nbayes_up<-auc(as.numeric(test_data$Class), as.numeric(pred_bayes_up))


##naive bayes for smote-sampling
set.seed(1)
smote_train$Class<-as.factor(smote_train$Class)
model_bayes_smote<-naive_bayes(Class~.,data = smote_train, usekernel = T)
pred_bayes_smote<-predict(model_bayes_smote,test_data)

## confusion matrix of naive bayes (smote-sampling data)
confusionMatrix(as.factor(pred_bayes_smote),as.factor(test_data$Class))
tp<-5655
tn<-10
fn<-1
fp<-31
precision<-tp/(tp+fp)
recall<-tp/(tp+fn)
f1_score<-(2*precision*recall)/(precision+recall)
tnr<-tn/(tn+fn)

auc_nbayes_smote<-auc(as.numeric(test_data$Class), as.numeric(pred_bayes_smote))

##naive bayes for rose-sampling
set.seed(1)
model_bayes_rose<-naive_bayes(Class~.,data = rose_train, usekernel = T)
pred_bayes_rose<-predict(model_bayes_rose,test_data)

## confusion matrix of naive bayes (rose sampling data)
confusionMatrix(as.factor(pred_bayes_rose),as.factor(test_data$Class))
tp<-5599
tn<-10
fn<-1
fp<-87
precision<-tp/(tp+fp)
recall<-tp/(tp+fn)
f1_score<-(2*precision*recall)/(precision+recall)
tnr<-tn/(tn+fn)

auc_nbayes_rose<-auc(as.numeric(test_data$Class), as.numeric(pred_bayes_rose))


## ROC of naive bayes for all samplings and imbalanced data
set.seed(1)
par(pty = "s")
roc(as.numeric(test_data$Class),as.numeric(pred_bayes_rose), col = "deepskyblue", plot = TRUE, main = "ROC: Naive Bayes", 
    xlab = "False positive rate", ylab = "True positive rate", lwd = 3,
    legacy.axes = TRUE)
plot(roc(as.numeric(test_data$Class),as.numeric(pred_bayes_smote),
         legacy.axes = TRUE, ), add = TRUE, col ="brown1", lwd = 3)
plot(roc(as.numeric(test_data$Class),as.numeric(pred_bayes_down),
         legacy.axes = TRUE),  add = TRUE, col = "chartreuse3", lwd = 3)
plot(roc(as.numeric(test_data$Class),as.numeric(pred_bayes_up),
         legacy.axes = TRUE), add = TRUE, col = "darkorchid1", lwd = 3)
plot(roc(as.numeric(test_data$Class),as.numeric(pred_bayes_rose),
         legacy.axes = TRUE), add = TRUE, col = "cyan3", lwd = 3)
legend(0.5,0.25, cex = 0.6, text.font = 1,legend=c("Imbalanced (AUC = 0.90)", "Down-sampling (AUC = 0.93)","Up-sampling (AUC = 0.95)",
                                                   "SMOTE (AUC = 0.95)","ROSE (AUC = 0.90)"),
       col=c("deepskyblue", "chartreuse3", "darkorchid1", "brown1", "cyan3"), lwd=3)
