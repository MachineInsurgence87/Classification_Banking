#DATA PREPARATION AND CLEANING OF DATA:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('/Users/deepakmathew/Downloads/archive/train.csv',sep=';') #import train data
train.head(3)
test = pd.read_csv('/Users/deepakmathew/Downloads/archive/test.csv',sep=';') #import test data
test.head(3) 

import seaborn as sns
import gc #garbage collector
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split #splitting the data for the training and test
from sklearn.linear_model import LogisticRegression #logistic regression modelling
from sklearn.metrics import classification_report #classification
from sklearn.metrics import confusion_matrix #confusion matrix after the graph
from sklearn.model_selection import cross_val_score #cross validating
from imblearn.over_sampling import RandomOverSampler #sampling of random values
from sklearn.metrics import roc_curve #roc curve analysis

train.info() #information about the dataset

train.isna().sum() #checking for null values

train.duplicated().sum() #checking the duplicates

train.duplicated().value_counts() #checking the duplicates

train.columns = train.columns.str.replace(' ','') #removal of the white spaces in the attributes
train.columns

train['job'].value_counts() #checking the counts of the unique values in the column attribute 'job

train['month'].value_counts()
train.corr() #correlation between the attributes. c<0.3:weak correlation, 0.3<c<0.7:moderately, 0.7<c<1.0:highly correlated
train['contact'].value_counts()

#Plotting of the categorical and numerical data:
train.columns
train.head(3)

#DATA PREPROCESSING, VISUALIZATION, DATA ANALYSIS:
#Categorical data analysis:
#Contact: Contact type such as cellular, telephone and  few are unknown:
#theme - figsize - plotting type - parameters angle - heading - execution

#Contact:
sns.set_theme(style = 'darkgrid') #theme
sns.set(rc = {'figure.figsize':(4,10)}) #figure size
contact = sns.countplot(x='contact', order=train['contact'].value_counts().index, data=train) #plot type assigning
contact.tick_params(axis='x', rotation = 60) #parameters rotation
plt.title('Bivariate analysis between contact and the y') #heading
plt.show() #execution

#Job:
train['job'].value_counts()
sns.set_theme(style = 'darkgrid') #theme
sns.set(rc = {'figure.figsize':(6,4)}) #figure size
job = sns.countplot(x='job', hue = 'y', order=train['job'].value_counts().index, data=train) #plot type assigning. y is the dependant variable
#hue plots will contain multiple bar containers
job.tick_params(axis='x', rotation = 60) #parameters rotation
plt.title('Bivariate analysis between job and the y') #heading
plt.show() #execution

#Marital:
train['marital'].value_counts()
sns.set_theme(style='darkgrid')
sns.set(rc = {'figure.figsize':(6,4)})
marital = sns.countplot(data = train, x = 'marital', hue='y', order = train['marital'].value_counts().index) #index is the index in the value counts
marital.tick_params(axis='x', rotation = 60)
plt.title('Bivariate analysis between marital and the y')
plt.show()

#Education:
train['education'].value_counts()

sns.set_theme(style = 'darkgrid')
sns.set(rc = {'figure.figsize':(6,4)})
education = sns.countplot(data = train, x='education', order = train['education'].value_counts().index)
education.tick_params(axis='x', rotation = 60)
education.bar_label(education.containers[0]) #if we are not using hue
plt.title("Bivariant analysis between education and y")
plt.show()

#Education with hue(the dependant variable)
sns.set_theme(style = 'darkgrid')
sns.set(rc = {'figure.figsize':(6,4)})
education = sns.countplot(data = train, x='education', hue='y', order = train['education'].value_counts().index)
education.tick_params(axis='x', rotation = 60)
for container in education.containers: #If we are using the hue
    education.bar_label(container)
plt.title("Bivariant analysis between education and y")
plt.show()


#Month plotting:
sns.set_theme(style = 'darkgrid')
sns.set(rc= {'figure.figsize' : (6,4)})
month = sns.countplot(data = train, x = 'month', hue = 'y', order =  train['month'].value_counts().index)
month.tick_params(axis = 'x', rotation = 60)
for i in month.containers:
    month.bar_label(i, rotation = 90) #here we have added the rotation of the index
plt.title('Bivariant comparision between month and y')
plt.show()


#Poutcome
sns.set_theme(style = 'darkgrid')
sns.set(rc = {'figure.figsize)':(6,4)})
poutcome = sns.countplot(data = train, x = 'poutcome', hue = 'y', order = train['poutcome'].value_counts().index)
poutcome.tick_params(axis = 'x', rotation = 60)
for i in poutcome.containers:
    poutcome.bar_label(i, rotation = 90) #containers are the values in the valuecounts index
plt.label('Bivariant analysis between poutcome and y')
plt.show()

#Numerical data analysis:

#Test for f prefix. f allows the expression inside a string to return
a = 'Deepak'
b = 10
c = 'Dubai'

print(f'Hello {a}, your age is {b}, and you stay in {c}')

train.columns
#Index(['age','default', 'balance',
       #'loan', 'day', 'duration', 'campaign', 'pdays',
       #'previous',

#Numerical data analysis:
#Age:
#Boxplot and histogram
#f in the prefix allows the expression inside a string to run.

#Syntax: plot initiation for subplots - theme - figure size - plotting - label - excecution
#mean -median - mode
#boxplot - plotting - lining axvline - label
#histogram - plotting - lining axvline - legends - label
#execution


#https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html 
#reference to the fig and axs usage while plotting subplots
fig, (ax_box, ax_hist) = plt.subplots(2, sharex = True, gridspec_kw= {"height_ratios": (0.3, 1)}) #height for the plot for 2 rows
sns.set_theme(style = 'whitegrid') #theme setting
sns.set(rc = {'figure.figsize': (11,8)}, font_scale = 1.5) #fontsize for the plot parameters
mean = train['age'].mean() #do not forget () for mean, median and mode
median = train['age'].median()
mode = train['age'].mode().values[0]

age = sns.boxplot(data = train, x ='age', y='y', order = train['y'].value_counts().index, ax=ax_box) #plotting
ax_box.axvline(mean, color = 'r', linestyle = '--') #axvline beacause we need to see the lines of analysis
ax_box.axvline(median, color = 'g', linestyle = '-')
ax_box.axvline(mode, color = 'b', linestyle = '-')


sns.histplot(data=train, x="age", ax=ax_hist, kde=True) #plotting histogram
ax_hist.axvline(mean, color='r', linestyle='--', label="Mean") #labelling and plotting lines
ax_hist.axvline(median, color='g', linestyle='-', label="Median")
ax_hist.axvline(mode, color='b', linestyle='-', label="Mode")
ax_hist.legend() #legend inside the graph

ax_box.set(xlabel='') #'' because to leave x axis parameter as blank
plt.show() #execute
#In age attribute the box plot focus mainly in the age betwee 30 and 50, probably because this age group is working people and with more
#salary. 

#balance:
fig, (ax_box, ax_hist) = plt.subplots(2, sharex = True, gridspec_kw= {'height_ratios':(0.5, 1)})
sns.set_theme(style = 'whitegrid')
sns.set(rc = {'figure.figsize': (11,8)})
mean = train['balance'].mean()
median = train['balance'].median()
mode = train['balance'].mode().values[0]

balance = sns.boxplot(data=train, x='balance', y='y', order = train['y'].value_counts().index, ax=ax_box)
ax_box.axvline(mean, color = 'r', linestyle = '--', label = 'mean')
ax_box.axvline(median, color='g', linestyle = '-', label = 'median')
ax_box.axvline(mode, color='b', linestyle = '-', label ='mode')
ax_box.set(xlabel='balance')
ax_box.legend()

sns.histplot(data=train, x='balance', ax = ax_hist) #kde = True
ax_hist.axvline(mean, color = 'r', linestyle = '--', label = 'mean')
ax_hist.axvline(median, color = 'g', linestyle = '-', label = 'median')
ax_hist.axvline(mode, color = 'b', linestyle = '-', label = 'mode')
ax_hist.set(xlabel = 'balance')
ax_hist.legend()
plt.show()
#Median is almost in 0. Most people contacted have nearly average 0 balance yearly.

#day:
fig, (ax_box, ax_hist) = plt.subplots(2, sharex = True, gridspec_kw= {'height_ratios':(0.5, 1)})
sns.set_theme(style = 'whitegrid')
sns.set(rc = {'figure.figsize': (11,8)})
mean = train['day'].mean()
median = train['day'].median()
mode = train['day'].mode().values[0]

loan = sns.boxplot(data=train, x='day', y='y', order = train['y'].value_counts().index, ax=ax_box)
ax_box.axvline(mean, color = 'r', linestyle = '--', label = 'mean')
ax_box.axvline(median, color='g', linestyle = '-', label = 'median')
ax_box.axvline(mode, color='b', linestyle = '-', label ='mode')
ax_box.set(xlabel='day')
ax_box.legend()

sns.histplot(data=train, x='day', ax = ax_hist, kde = True) #kde will show the trend line
ax_hist.axvline(mean, color = 'r', linestyle = '--', label = 'mean')
ax_hist.axvline(median, color = 'g', linestyle = '-', label = 'median')
ax_hist.axvline(mode, color = 'b', linestyle = '-', label = 'mode')
ax_hist.set(xlabel = 'day')
ax_hist.legend()
plt.show()
#the histogram shows symmetry almost in the graph but peak on the day 20. This feature is not much useful as there is no much output

#duration:
fig, (ax_box, ax_hist) = plt.subplots(2, sharex = True, gridspec_kw= {'height_ratios':(0.5, 1)})
sns.set_theme(style = 'whitegrid')
sns.set(rc = {'figure.figsize': (11,8)})
mean = train['duration'].mean()
median = train['duration'].median()
mode = train['duration'].mode().values[0]

loan = sns.boxplot(data=train, x='duration', y='y', order = train['y'].value_counts().index, ax=ax_box)
ax_box.axvline(mean, color = 'r', linestyle = '--', label = 'mean')
ax_box.axvline(median, color='g', linestyle = '-', label = 'median')
ax_box.axvline(mode, color='b', linestyle = '-', label ='mode')
ax_box.set(xlabel='duration')
ax_box.legend()

sns.histplot(data=train, x='duration', ax = ax_hist, kde = True) #kde will show the trend line
ax_hist.axvline(mean, color = 'r', linestyle = '--', label = 'mean')
ax_hist.axvline(median, color = 'g', linestyle = '-', label = 'median')
ax_hist.axvline(mode, color = 'b', linestyle = '-', label = 'mode')
ax_hist.set(xlabel = 'duration')
ax_hist.legend()
plt.show()
#When the call spent was more than the average value, the outcome was more. This feature has an effect in 'y'.

#campaign
fig, (ax_box, ax_hist) = plt.subplots(2, sharex = True, gridspec_kw= {'height_ratios':(0.5, 1)})
sns.set_theme(style = 'whitegrid')
sns.set(rc = {'figure.figsize': (11,8)})
mean = train['campaign'].mean()
median = train['campaign'].median()
mode = train['campaign'].mode().values[0]

loan = sns.boxplot(data=train, x='campaign', y='y', order = train['y'].value_counts().index, ax=ax_box)
ax_box.axvline(mean, color = 'r', linestyle = '--', label = 'mean')
ax_box.axvline(median, color='g', linestyle = '-', label = 'median')
ax_box.axvline(mode, color='b', linestyle = '-', label ='mode')
ax_box.set(xlabel='campaign')
ax_box.legend()

sns.histplot(data=train, x='campaign', ax = ax_hist, kde = True) #kde will show the trend line
ax_hist.axvline(mean, color = 'r', linestyle = '--', label = 'mean')
ax_hist.axvline(median, color = 'g', linestyle = '-', label = 'median')
ax_hist.axvline(mode, color = 'b', linestyle = '-', label = 'mode')
ax_hist.set(xlabel = 'campaign')
ax_hist.legend()
plt.show()
#Here the yes and no are almost same. The campaign is the number of times contact was made. The mode is almost close to 0, therefore
#if we contact more, there is possibly a waste of time. Instead the less contacts made people more in investment

#pdays;
fig, (ax_box, ax_hist) = plt.subplots(2, sharex = True, gridspec_kw= {'height_ratios':(0.5, 1)})
sns.set_theme(style = 'whitegrid')
sns.set(rc = {'figure.figsize': (11,8)})
mean = train['pdays'].mean()
median = train['pdays'].median()
mode = train['pdays'].mode().values[0]
print(mean, median, mode) #for printable results

loan = sns.boxplot(data=train, x='pdays', y='y', order = train['y'].value_counts().index, ax=ax_box)
ax_box.axvline(mean, color = 'r', linestyle = '--', label = 'mean')
ax_box.axvline(median, color='g', linestyle = '-', label = 'median')
ax_box.axvline(mode, color='b', linestyle = '-', label ='mode')
ax_box.set(xlabel='pdays')
ax_box.legend()

sns.histplot(data=train, x='pdays', ax = ax_hist, kde = True) #kde will show the trend line
ax_hist.axvline(mean, color = 'r', linestyle = '--', label = 'mean')
ax_hist.axvline(median, color = 'g', linestyle = '-', label = 'median')
ax_hist.axvline(mode, color = 'b', linestyle = '-', label = 'mode')
ax_hist.set(xlabel = 'pdays')
ax_hist.legend()
plt.show()
#number of days that passed by after the client was last contacted from a previous campaign
#here the mode and median are -1. Therefore most of the people were contacted only once. This feature will be removed due to lack of impact.
#and -1 indicates that the clients were not previously contacted

#previous
fig, (ax_box, ax_hist) = plt.subplots(2, sharex = True, gridspec_kw= {'height_ratios':(0.5, 1)})
sns.set_theme(style = 'whitegrid')
sns.set(rc = {'figure.figsize': (11,8)})
mean = train['previous'].mean()
median = train['previous'].median()
mode = train['previous'].mode().values[0]
print(mean, median, mode) #for printable results

loan = sns.boxplot(data=train, x='previous', y='y', order = train['y'].value_counts().index, ax=ax_box)
ax_box.axvline(mean, color = 'r', linestyle = '--', label = 'mean')
ax_box.axvline(median, color='g', linestyle = '-', label = 'median')
ax_box.axvline(mode, color='b', linestyle = '-', label ='mode')
ax_box.set(xlabel='previous')
ax_box.legend()

sns.histplot(data=train, x='previous', ax = ax_hist, kde = True) #kde will show the trend line
ax_hist.axvline(mean, color = 'r', linestyle = '--', label = 'mean')
ax_hist.axvline(median, color = 'g', linestyle = '-', label = 'median')
ax_hist.axvline(mode, color = 'b', linestyle = '-', label = 'mode')
ax_hist.set(xlabel = 'previous')
ax_hist.legend()
plt.show()
#number of days that passed by after the client was last contacted from a previous campaign. There is no much information.
#Hence this feature would be removed.

#FEATURE SELECTION:
#According to the previous analysis the following features are selected to implement a logistic regression model:
    #Categorical features: ["job","marital","education","default","housing", "loan", "contact", "poutcome", "y"]
    #Numerical features: ["age", "balance", "duration", "campaign"]

#Concatenating the train test data:
train_copy = train.copy() #Creating a copy of train data
test_copy = test.copy() #Creating a copy of test data
train_copy['tst'] = 0 #Assigning extra column to show the data based on train data
test_copy['tst'] = 1 #Assigning extra column to show the data based on test data
train_test_concat = pd.concat([train_copy, test_copy], ignore_index = True)
del train_copy
del test_copy
gc.collect() #usually ran to collect the unused data and free the memory

print(train_test_concat.shape) #(49732, 18)
print(train.shape) #(45211, 17)
print(test.shape) #(4521, 17)

#Filling the required records for each feature
train['job']. value_counts() #There is a record as 'unknown' which has to be filled by mode
train_test_concat['job'].replace(['unknown'], train_test_concat['job'].mode(), inplace = True)

train['education'].value_counts()
train_test_concat['education'].replace(['unknown'], train_test_concat['education'].mode(), inplace = True)

train_test_concat['contact'].value_counts()
train_test_concat['contact'].replace(['unknown'], train_test_concat['contact'].mode(), inplace = True)

#Drop unnecessary features:
train_test_concat.columns
train_test_concat.drop(columns = ['month', 'previous', 'day', 'pdays'], inplace = True)
train_test_concat.head(5)
train_test_concat.shape
train_test_concat['contact'].value_counts()
#Encoding categorical features: encodes categorical variables into 1s and 0s (yes and no become 1 and 0). 
#When the feature has more than two variables to encode: "marital": married, single, divorced become [1,0,0],[0,1,0], [0,0,1]. 
#For this case is used pd.get_dummies(train["Column_names_here"])

#Encoding:
#Categorical data: job, marital, education, poutcome, default, loan, housing
#Data with 2 records: default, loan, housing, contact, y
train_test_concat['default'] = train_test_concat['default'].map({'yes':1, 'no':0})
train_test_concat['loan'] = train_test_concat['loan'].map({'yes':1, 'no':0})
train_test_concat['housing'] = train_test_concat['housing'].map({'yes':1, 'no':0})
train_test_concat['contact'] = train_test_concat['contact'].map({'telephone':1, 'cellular':0})
train_test_concat['y'] = train_test_concat['y'].map({'yes':1, 'no':0})

train_test_concat.head(5)

#Categorical data with more than 2 records: job, marital, education, poutcome
#Here we use the dummies system
train_test_concat = pd.get_dummies(train_test_concat, columns=['job', 'marital', 'education', 'poutcome'])
train_test_concat.columns
train_test_concat.dtypes
train_test_concat.shape

#LOGISTIC REGRESSION MODEL FOR CLASSIFICATION:
#Logistic regression method is used because we need to get the final answer as either yes or no and also to perform the binary classification

X = train_test_concat.drop('y', axis = 1)
y = train_test_concat['y']
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, test_size = 0.4, random_state = 49) #Data splitting

#logistic regression model 1:
logreg = LogisticRegression() #assigning logreg as logistic regression model
logreg.fit(Xtrain, ytrain) #fitting the training data with logistic regression
y_pred = logreg.predict(Xtest) #predicting the y variable in the test data by using X values of test data
y_pred

#Evaluation of regression model 1:
print(confusion_matrix(y_pred, ytest))     
            #   precision    recall  f1-score   support

           #0       0.91      0.98      0.94     17550
           #1       0.62      0.27      0.38      2343

    #accuracy                           0.89     19893
   #macro avg       0.76      0.63      0.66     19893
#weighted avg       0.88      0.89      0.88     19893

print(classification_report(ytest, y_pred)) #used for the prediction


print('Accuracy:', metrics.accuracy_score(ytest, y_pred))
print('Precision:', metrics.precision_score(ytest, y_pred))
print('Recall:', metrics.recall_score(ytest, y_pred))
#cross validation: cross_val_score(log_reg, X_train_imputed, y_train, cv=10)
#here we are doing cross validation mean scoring
print('Cross validation mean:', cross_val_score(logreg, Xtrain, ytrain, cv = 10, n_jobs = 2, scoring = 'accuracy').mean())

#Here we can see 91% of correct predictions while the ability to catch positive cases are 27% that were correctly identified.
#38% are positive predictions. There is an imbalance and we need to work again with random oversampling method as below.

#Data is imbalanced, as the classification data is skewed. Classes that makes large portion of dataset is Majority class and 
#other is minority class.
#Random over sampler:
ROS  =RandomOverSampler(sampling_strategy='minority', random_state = 10)
#Here we are using sampling_strategy argument that can be set to “minority” to,
#automatically balance the minority class with majority class or classes
XtrainROS, ytrainROS = ROS.fit_resample(Xtrain, ytrain) #resampling for better shuffle
np.bincount(ytrainROS) #Count number of occurrences of each value in array of non-negative ints.

#logistic regression model 2:
logreg_ROS = LogisticRegression()
logreg_ROS.fit(XtrainROS, ytrainROS) #fitting the logreg to the new ROSampled X and y values
y_pred_ROS = logreg_ROS.predict(Xtest) #predicting based on the new RO sampled logreg

#Evaluation of regression model 2:
print(confusion_matrix(y_pred_ROS, ytest))
print(classification_report(ytest, y_pred_ROS))
print("Accuracy:",metrics.accuracy_score(ytest, y_pred_ROS))
print("Precision:",metrics.precision_score(ytest, y_pred_ROS))
print("Recall:",metrics.recall_score(ytest, y_pred_ROS))
print('Cross Validation mean:',(cross_val_score(logreg_ROS, Xtrain, ytrain, cv=5, n_jobs=2).mean()))
#Here the ability to return correct prediction is 96%, the ability to return positive cases is 76% and the ability to return
#positive predictions is 50%. Ability for better prediction got increased.

#ROC curve can be used to select a threshold for a classifier, which maximizes the true positives and in turn minimizes the false positives. 
#ROC curve is a graph used to show the performance of the classification model at all classification thresholds.
#Threshold is the intensity for a certain phoenomenon to occur.
#This curve has 2 parameters: TPR and FPR(Sensitivity and specificity)

#dependant variable test value's probability - figure size - using 1st value, assign x and y - auc - 
# - plot pyplot x vs y and legend - legend for the meaning of plot - plot execution
y_pred_proba = logreg_ROS.predict_proba(Xtest)[:,1]
sns.set(rc = {'figure.figsize':(6,4)})
fpr, tpr, _ = metrics.roc_curve(ytest, y_pred_proba)
auc = metrics.roc_auc_score(ytest, y_pred_proba)
plt.plot(fpr, tpr, label = 'data, auc='+str(auc))
plt.legend(loc = 'best')
plt.show()
print(metrics.roc_auc_score(ytest, y_pred_proba))
#AUC score is 0.87 here. The perfect classifier shall be shown in a score of 1.0
#It is highlighted that the inclusion of oversampling gives substantial improvements to the -
#Logistic Regression model with an accuracy of 82%. The recall presents 76% of positives cases -
#that were correctly identified which has proved the best method to overcome imbalance data.
# Also, it was achieved a cross-validation score of 89%.





