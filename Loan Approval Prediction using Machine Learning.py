# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# dependencies = [
#   "seaborn",
# ]
# ///
# MAGIC %md
# MAGIC ## **Loan Approval Prediction using Machine Learning**

# COMMAND ----------

# MAGIC %md
# MAGIC Loans are a major requirement of the modern world. By this alone, banks receive a major portion of the total profit. It is beneficial for students to manage their education and living expenses, and for individuals to purchase various luxuries, such as houses and cars. But when it comes to deciding whether the applicant's profile is relevant to be granted with loan or not. Banks have to look after many aspects.
# MAGIC
# MAGIC So, here we will be using machine learning algorithms to ease their work and predict whether the candidate’s profile is relevant or not, using key features like Marital Status, Education, Applicant Income, Credit History, etc.

# COMMAND ----------

# MAGIC %md
# MAGIC **Importing Libraries and Dataset**

# COMMAND ----------

# MAGIC %pip install seaborn 

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = spark.read.table("workspace.default.loan_approval_prediction").toPandas()

# COMMAND ----------

data.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC **Data Preprocessing and Visualization**

# COMMAND ----------

# DBTITLE 1,Get the number of columns of object datatype.
obj = (data.dtypes == 'object')
print("Categorical variables:",len(list(obj[obj].index)))

# COMMAND ----------

# MAGIC %md
# MAGIC As Loan_ID is completely unique and not correlated with any of the other column, So we will drop it using .drop() function.

# COMMAND ----------

# Dropping Loan_ID column
data.drop(['Loan_ID'],axis=1,inplace=True,errors='ignore')

# COMMAND ----------

# MAGIC %md
# MAGIC Visualize all the unique values in columns using barplot. This will simply show which value is dominating as per our dataset.

# COMMAND ----------

obj = (data.dtypes == 'object')
object_cols = list(obj[obj].index)
plt.figure(figsize=(18,36))
index = 1

for col in object_cols:
  y = data[col].value_counts()
  plt.subplot(11,4,index)
  plt.xticks(rotation=90)
  sns.barplot(x=list(y.index), y=y)
  index +=1

# COMMAND ----------

# MAGIC %md
# MAGIC As all the categorical values are binary so we can use Label Encoder for all such columns and the values will change into int datatype.

# COMMAND ----------

# Import label encoder
from sklearn import preprocessing
  
# label_encoder object knows how 
# to understand word labels.
label_encoder = preprocessing.LabelEncoder()
obj = (data.dtypes == 'object')
for col in list(obj[obj].index):
  data[col] = label_encoder.fit_transform(data[col])

# COMMAND ----------

# To find the number of columns with 
# datatype==object
obj = (data.dtypes == 'object')
print("Categorical variables:",len(list(obj[obj].index)))

# COMMAND ----------

plt.figure(figsize=(12,6))

sns.heatmap(data.corr(),cmap='BrBG',fmt='.2f',
            linewidths=2,annot=True)

# COMMAND ----------

# MAGIC %md
# MAGIC The above heatmap is showing the correlation between Loan Amount and ApplicantIncome. It also shows that Credit_History has a high impact on Loan_Status.

# COMMAND ----------

#we will use Catplot to visualize the plot for the Gender, and Marital Status of the applicant.
 
sns.catplot(x="Gender", y="Married",
            hue="Loan_Status", 
            kind="bar", 
            data=data)

# COMMAND ----------

#we will find out if there is any missing values in the dataset using below code.
for col in data.columns:
  data[col] = data[col].fillna(data[col].mean()) 
  
data.isna().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC Splitting Dataset 

# COMMAND ----------

from sklearn.model_selection import train_test_split

X = data.drop(['Loan_Status'],axis=1)
Y = data['Loan_Status']
X.shape,Y.shape

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.4,
                                                    random_state=1)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

# COMMAND ----------

# MAGIC %md
# MAGIC **Model Training and Evaluation**

# COMMAND ----------

# MAGIC %md
# MAGIC As this is a classification problem so we will be using these models : 
# MAGIC
# MAGIC - KNeighborsClassifiers
# MAGIC - RandomForestClassifiers
# MAGIC - Support Vector Classifiers (SVC)
# MAGIC - Logistics Regression

# COMMAND ----------

#To predict the accuracy we will use the accuracy score function from scikit-learn library.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn import metrics

knn = KNeighborsClassifier(n_neighbors=3)
rfc = RandomForestClassifier(n_estimators = 7,
                             criterion = 'entropy',
                             random_state =7)
svc = SVC()
lc = LogisticRegression()

# making predictions on the training set
for clf in (rfc, knn, svc,lc):
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_train)
    print("Accuracy score of ",
          clf.__class__.__name__,
          "=",100*metrics.accuracy_score(Y_train, 
                                         Y_pred))

# COMMAND ----------


# making predictions on the testing set
for clf in (rfc, knn, svc,lc):
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    print("Accuracy score of ",
          clf.__class__.__name__,"=",
          100*metrics.accuracy_score(Y_test,
                                     Y_pred))