#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as ex
import missingno as msno
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix,roc_curve,roc_auc_score


# In[2]:


df=pd.read_csv('C:/Users/Sumit/Downloads/Loan_Default.csv')


# In[3]:


df


# In[4]:


pd.set_option("display.max_columns",None)


# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


df.columns


# ID = Customer ID of Applicant
# 
# year = Year of Application
# 
# loan limit = maximum avaliable amount of the loan allowed to be taken
# 
# Gender = sex type
# 
# approv_in_adv = Is loan pre-approved or not
# 
# loan_type = Type of loan
# 
# loan_purpose = the reason you want to borrow money
# 
# Credit_Worthiness = is how a lender determines that you will default on your debt obligations, or how worthy you are to receive new credit.
# 
# open_credit = is a pre-approved loan between a lender and a borrower. It allows the borrower to make repeated withdrawals up to a certain limit.
# 
# business_or_commercial = Usage type of the loan amount
# 
# loan_amount = The exact loan amount
# 
# rate_of_interest = is the amount a lender charges a borrower and is a percentage of the principal—the amount loaned.
# 
# Interest_rate_spread = the difference between the interest rate a financial institution pays to depositors and the interest rate it receives from loans
# 
# Upfront_charges = Fee paid to a lender by a borrower as consideration for making a new loan
# 
# term = the loan's repayment period
# 
# Neg_ammortization = refers to a situation when a loan borrower makes a payment less than the standard installment set by the bank.
# 
# interest_only = amount of interest only without principles
# 
# lump_sum_payment = is an amount of money that is paid in one single payment rather than in installments.
# 
# property_value = the present worth of future benefits arising from the ownership of the property
# 
# construction_type = Collateral construction type
# 
# occupancy_type = classifications refer to categorizing structures based on their usage
# 
# Secured_by = Type of Collatoral
# 
# total_units = number of unites
# 
# income = refers to the amount of money, property, and other transfers of value received over a set period of time
# 
# credit_type = type of credit
# 
# co-applicant_credit_type = is an additional person involved in the loan application process. Both applicant and co-applicant apply and sign for the loan
# 
# age = applicant's age
# 
# submission_of_application = Ensure the application is complete or not
# 
# LTV = life-time value (LTV) is a prognostication of the net profit
# 
# Region = applicant's place
# 
# Security_Type = Type of Collatoral
# 
# status = Loan status (Approved/Declined)
# 
# dtir1 = debt-to-income ratio

# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


ex.pie(df,names='Status',title='Proportion of Loan Status')


# In[11]:


sns.countplot(x='Gender',data=df)


# In[12]:


sns.countplot(x='Gender',hue='Status',data=df,palette='Set1')


# In[13]:


sns.countplot(x='Region',hue='Status',data=df,palette='Set2')


# In[14]:


sns.countplot(x='Secured_by',hue='Status',data=df,palette='Set1')


# In[15]:


sns.catplot(data=df,x='age',y='loan_amount',kind='box',col='Status',sym='')


# In[16]:


fig,axes = plt.subplots(3,2, figsize=(25,20))

sns.countplot('open_credit',hue='Status',data=df,ax=axes[0,0],palette='Set1')
sns.countplot('business_or_commercial',hue='Status',data=df,ax=axes[1,0],palette='Set1')
sns.countplot('Neg_ammortization',hue='Status',data=df,ax=axes[0,1],palette='Set1')
sns.countplot('age',hue='Status',data=df,ax=axes[1,1],palette='Set1')
sns.countplot('Region',hue='Status',data=df,ax=axes[2,0],palette='Set1')
sns.countplot('Gender',hue='Status',data=df,ax=axes[2,1],palette='Set1')


# In[17]:


df.isna().sum()


# # Plotting the NULL values distribution for each column

# In[18]:


msno.matrix(df)
plt.show()


# In[19]:


df['rate_of_interest']= df['rate_of_interest'].fillna(df['rate_of_interest'].mean())
df['Upfront_charges']= df['Upfront_charges'].fillna(df['Upfront_charges'].mean())
df['term']= df['term'].fillna(df['term'].mean())
df['property_value']= df['property_value'].fillna(df['property_value'].median())
df['income']= df['income'].fillna(df['income'].median())
df['LTV']= df['LTV'].fillna(df['LTV'].mean())
df['dtir1']= df['dtir1'].fillna(df['dtir1'].mean())


# In[20]:


df.isna().sum()


# In[21]:


df_no_na=df.dropna()


# In[22]:


df_no_na.shape


# As only (148670-147441)= 1229 rows get deleted after deleting all the NULL values corresponding to categorical variables. It is reasonable as the total data points is 1.5 lakh

# # Checking for NULL values again

# In[23]:


df_no_na.isna().sum()


# In[24]:


msno.matrix(df_no_na)
plt.show()


# In[25]:


ex.pie(df_no_na,names='Gender',title='Gender pie plot',hole=0.3)


# In[26]:


ex.pie(df_no_na,names='open_credit',title='Pie chart of open credit',hole=0.3)


# In[27]:


ex.pie(df_no_na,names='business_or_commercial',title='Purpose of the loan',hole=0.3)


# In[28]:


ex.pie(df_no_na,names='Neg_ammortization',title='Neg ammortization pie plot',hole=0.3)


# In[29]:


ex.pie(df_no_na,names='age',title='Age pie plot',hole=0.3)


# In[30]:


ex.pie(df_no_na,names='Region',title='Region pie plot',hole=0.3)


# In[31]:


df_no_na.nunique()


# ID and year are dropped

# In[32]:


df1=df_no_na.drop(['ID','year','Upfront_charges'],axis=1)


# In[33]:


df1.columns


# In[34]:


df1.corr()


# In[35]:


sns.heatmap(df1.corr(),annot=True)


# There is not as such high correlation between all the regressor variables.

# # Removing the Outliers

# In[36]:


outliers=[]
def detect_outlier(data_1):
    
    threshold=3
    mean_1 = np.mean(data_1)
    std_1 =np.std(data_1)
    
    
    for y in data_1:
        z_score=(y - mean_1)/std_1
        if np.abs(z_score) > threshold:            outliers.append(y)
    return outliers        


# In[37]:


outlier_datapoints = detect_outlier(df1['LTV'])
print(outlier_datapoints)


# In[38]:


sns.set(rc={"figure.figsize":(18,5)})

sns.boxplot(df1['LTV'],palette='summer')


# In[39]:


outlier_datapoints = detect_outlier(df1['Credit_Score'])
print(outlier_datapoints)


# In[40]:


sns.set(rc={"figure.figsize":(30,5)})

sns.boxplot(df1['Credit_Score'],palette='summer')


# In[41]:


outlier_datapoints = detect_outlier(df1['dtir1'])
print(outlier_datapoints)


# In[42]:


sns.set(rc={"figure.figsize":(30,5)})

sns.boxplot(df1['dtir1'],palette='summer')


# Credit score generally lies within 850.So, I have deleted some outliers.

# In[43]:


df1.drop(df1.loc[df1['Credit_Score']>=900].index, inplace=True)


# In[44]:


df1.drop(df1.loc[df1['LTV']>2000].index,inplace=True)


# # Taking the group of categorical variable

# In[45]:


df1_categorical=df1.select_dtypes(include=['object'])


# In[46]:


df1_categorical.head()


# # Getting the dummy variable for the categorical variables

# In[47]:


df1_cat_dummy =pd.get_dummies(df1_categorical,drop_first=True)


# In[48]:


df1_cat_dummy.head()


# In[49]:


df1_cat_dummy.rename(columns = {'age_<25':'age_less_than_25' ,'age_>74':'age_greater_than_74'},inplace = True)


# Dropping the categorical variables

# In[50]:


df2=df1.drop(df1_categorical.columns, axis=1)


# In[51]:


df2.head()


# In[52]:


sns.pairplot(df2)


# In[53]:


Target_var=df2['Status']
numeric_unscaled=df2.drop('Status',axis=1)


# In[54]:


numeric_unscaled.head()


# # Multicollinearity

# In[55]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[56]:


def calc_vif(x):
    #calculating VIF
    vif=pd.DataFrame()
    vif['variables']=x.columns
    vif['VIF']=[variance_inflation_factor(x.values,i) for i in range(x.shape[1])]
    
    return(vif)


# In[57]:


calc_vif(numeric_unscaled)


# VIF value of rate of interest is very high, so we will drop it.

# In[58]:


numeric_unscaled_f=numeric_unscaled.drop('rate_of_interest',axis=1)


# In[60]:


calc_vif(numeric_unscaled_f)


# In[61]:


fig,axes =plt.subplots(nrows=3,ncols=3,figsize=(15,7),sharex=False,sharey=False)
axes=axes.ravel()
cols =numeric_unscaled_f.columns[:]

for col,ax in zip(cols,axes):
    
    
    sns.kdeplot(data=numeric_unscaled_f, x=col, shade=True, ax=ax)
    ax.set(title='f Distribution of Variable: {col}', xlabel=None)
    
    
    
fig.delaxes(axes[7])
fig.tight_layout()
plt.show()


# In[62]:


numeric_unscaled_f.describe()


# In[63]:


Target_var.head()


# In[64]:


Target_var.shape


# # Scaling the numerical data

# In[69]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
numeric_scaled=pd.DataFrame(sc.fit_transform(numeric_unscaled_f))
numeric_scaled.columns=numeric_unscaled_f.columns


# In[70]:


numeric_scaled


# In[71]:


numeric_scaled.shape


# In[72]:


df1_cat_dummy.shape


# In[73]:


numeric_scaled=numeric_scaled.reset_index(drop=True)


# In[74]:


df1_cat_dummy=df1_cat_dummy.reset_index(drop=True)


# In[75]:


x_scaled=pd.concat([numeric_scaled,df1_cat_dummy],axis='columns')


# In[76]:


x_scaled.shape


# # Splitting the data

# In[77]:


x_train,x_test,y_train,y_test=train_test_split(x_scaled,Target_var,test_size=0.3,random_state=0,stratify=Target_var)


# In[78]:


len(x_train)


# In[79]:


len(x_test)


# In[80]:


x_train.head()


# In[81]:


y_train.head()


# # Balancing the data

# In[82]:


sns.countplot(x='Status',data=df_no_na)


# In[83]:


ex.pie(df_no_na,names='Status',title='Proportion of Loan Status')


# In the dataset, around 75% data are of not default loan and only 25% of the data is showing as defaulter, so the data is unbalanced. We have to balance it.

# In[84]:


from imblearn import under_sampling,over_sampling
from imblearn.over_sampling import SMOTE
from collections import Counter


# In[85]:


Oversample=SMOTE(sampling_strategy=1,random_state=42,k_neighbors=3)
x_train_over,y_train_over=Oversample.fit_resample(x_train,y_train)
print('Distribution target class after oversample:', Counter(y_train_over))


# # Logistic Regression

# In[86]:


from sklearn.linear_model import LogisticRegression


# In[87]:


lr = LogisticRegression(max_iter=500)
lr.fit(x_train,y_train)
y_test_pred_lr=lr.predict(x_test)


# In[88]:


Accuracy=accuracy_score(y_test,y_test_pred_lr)
Precision=precision_score(y_test,y_test_pred_lr)
Recall=recall_score(y_test,y_test_pred_lr)
F1_score=f1_score(y_test,y_test_pred_lr)
y_test_pred_prob=lr.predict_proba(x_test)[:,1]
AUC=roc_auc_score(y_test,y_test_pred_prob)

print("========== Model Performance =============")
print("Test Accuracy: %0.3f%%" %(Accuracy*100.0))
print("Precision: %0.3f%%" %(Precision*100.0))
print("Recall: %0.3f%%" %(Recall*100.0))
print("F1_score: %0.3f%%" %(F1_score*100.0))
print("AUC Score: %0.3f%%" %(AUC*100.0))
print("*******************************")
plt.figure(figsize=(3,3))
sns.heatmap(confusion_matrix(y_test,y_test_pred_lr), annot=True, cmap="YlGnBu",fmt='g',cbar=False)
plt.title("Test Confusion Matrix",fontsize=16,fontweight="bold")
plt.show()


# Using Balanced Data

# In[89]:


lr = LogisticRegression(max_iter=500)
lr.fit(x_train_over,y_train_over)
y_test_pred_lr=lr.predict(x_test)


# In[90]:


Accuracy=accuracy_score(y_test,y_test_pred_lr)
Precision=precision_score(y_test,y_test_pred_lr)
Recall=recall_score(y_test,y_test_pred_lr)
F1_score=f1_score(y_test,y_test_pred_lr)
y_test_pred_prob=lr.predict_proba(x_test)[:,1]
AUC=roc_auc_score(y_test,y_test_pred_prob)

print("========== Model Performance =============")
print("Test Accuracy: %0.3f%%" %(Accuracy*100.0))
print("Precision: %0.3f%%" %(Precision*100.0))
print("Recall: %0.3f%%" %(Recall*100.0))
print("F1_score: %0.3f%%" %(F1_score*100.0))
print("AUC Score: %0.3f%%" %(AUC*100.0))
print("*******************************")
plt.figure(figsize=(3,3))
sns.heatmap(confusion_matrix(y_test,y_test_pred_lr), annot=True, cmap="YlGnBu",fmt='g',cbar=False)
plt.title("Test Confusion Matrix",fontsize=16,fontweight="bold")
plt.show()


# Hyper-parameter-tuning

# In[91]:


from sklearn.model_selection import GridSearchCV


# In[93]:


#grid search cross validation
grid={"C":np.logspace(-3,3,7), "penalty":['l1','l2'],'max_iter':[100,200,300,400,500,1000]} #l1 lasso l2 ridge
lr_cv=GridSearchCV(lr,grid,cv=5)
lr_cv.fit(x_train,y_train)

print('tuned hyperparameters :(best parameters)',lr_cv.best_params_)


# In[97]:


lr2 = LogisticRegression(C=1000.0, max_iter=500, penalty = 'l2')
lr2.fit(x_train_over,y_train_over)
y_test_pred_lr2=lr2.predict(x_test)


# In[98]:


Accuracy=accuracy_score(y_test,y_test_pred_lr2)
Precision=precision_score(y_test,y_test_pred_lr2)
Recall=recall_score(y_test,y_test_pred_lr2)
F1_score=f1_score(y_test,y_test_pred_lr2)
y_test_pred_prob=lr2.predict_proba(x_test)[:,1]
AUC=roc_auc_score(y_test,y_test_pred_prob)

print("========== Model Performance =============")
print("Test Accuracy: %0.3f%%" %(Accuracy*100.0))
print("Precision: %0.3f%%" %(Precision*100.0))
print("Recall: %0.3f%%" %(Recall*100.0))
print("F1_score: %0.3f%%" %(F1_score*100.0))
print("AUC Score: %0.3f%%" %(AUC*100.0))
print("*******************************")
plt.figure(figsize=(3,3))
sns.heatmap(confusion_matrix(y_test,y_test_pred_lr2), annot=True, cmap="YlGnBu",fmt='g',cbar=False)
plt.title("Test Confusion Matrix",fontsize=16,fontweight="bold")
plt.show()


# # KNN

# In[99]:


from sklearn.neighbors import KNeighborsClassifier


# In[100]:


knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train,y_train)
y_test_pred_knn=knn.predict(x_test)


# In[101]:


Accuracy=accuracy_score(y_test,y_test_pred_knn)
Precision=precision_score(y_test,y_test_pred_knn)
Recall=recall_score(y_test,y_test_pred_knn)
F1_score=f1_score(y_test,y_test_pred_knn)
y_test_pred_prob=knn.predict_proba(x_test)[:,1]
AUC=roc_auc_score(y_test,y_test_pred_prob)

print("========== Model Performance =============")
print("Test Accuracy: %0.3f%%" %(Accuracy*100.0))
print("Precision: %0.3f%%" %(Precision*100.0))
print("Recall: %0.3f%%" %(Recall*100.0))
print("F1_score: %0.3f%%" %(F1_score*100.0))
print("AUC Score: %0.3f%%" %(AUC*100.0))
print("*******************************")
plt.figure(figsize=(3,3))
sns.heatmap(confusion_matrix(y_test,y_test_pred_knn), annot=True, cmap="YlGnBu",fmt='g',cbar=False)
plt.title("Test Confusion Matrix",fontsize=16,fontweight="bold")
plt.show()


# Using Balanced Data

# In[102]:


knn1 = KNeighborsClassifier(n_neighbors=7)
knn1.fit(x_train_over,y_train_over)
y_test_pred_knn1=knn1.predict(x_test)


# In[103]:


Accuracy=accuracy_score(y_test,y_test_pred_knn1)
Precision=precision_score(y_test,y_test_pred_knn1)
Recall=recall_score(y_test,y_test_pred_knn1)
F1_score=f1_score(y_test,y_test_pred_knn1)
y_test_pred_prob=knn.predict_proba(x_test)[:,1]
AUC=roc_auc_score(y_test,y_test_pred_prob)

print("========== Model Performance =============")
print("Test Accuracy: %0.3f%%" %(Accuracy*100.0))
print("Precision: %0.3f%%" %(Precision*100.0))
print("Recall: %0.3f%%" %(Recall*100.0))
print("F1_score: %0.3f%%" %(F1_score*100.0))
print("AUC Score: %0.3f%%" %(AUC*100.0))
print("*******************************")
plt.figure(figsize=(3,3))
sns.heatmap(confusion_matrix(y_test,y_test_pred_knn1), annot=True, cmap="YlGnBu",fmt='g',cbar=False)
plt.title("Test Confusion Matrix",fontsize=16,fontweight="bold")
plt.show()


# Hyper-parameter-tuning

# grid={"n_neighbors":[3,5,1,19], "weights":['uniform','distance'],'metric':['euclidean','manhattan']} 
# knn_cv=GridSearchCV(knn,grid,cv=3,verbose=1,n_jobs=-1)
# knn_cv.fit(x_train,y_train)
# 
# print('tuned hyperparameters :(best parameters)',knn_cv.best_params_)

# knn2 = KNeighborsClassifier(C=1000.0, max_iter=500, penalty = 'l2')
# knn2.fit(x_train,y_train)
# y_test_pred_knn2=knn2.predict(x_test)

# Accuracy=accuracy_score(y_test,y_test_pred_knn2)
# Precision=precision_score(y_test,y_test_pred_knn2)
# Recall=recall_score(y_test,y_test_pred_knn2)
# F1_score=f1_score(y_test,y_test_pred_knn2)
# y_test_pred_prob=knn2.predict_proba(x_test)[:,1]
# AUC=roc_auc_score(y_test,y_test_pred_prob)
# 
# print("========== Model Performance =============")
# print("Test Accuracy: %0.3f%%" %(Accuracy*100.0))
# print("Precision: %0.3f%%" %(Precision*100.0))
# print("Recall: %0.3f%%" %(Recall*100.0))
# print("F1_score: %0.3f%%" %(F1_score*100.0))
# print("AUC Score: %0.3f%%" %(AUC*100.0))
# print("*******************************")
# plt.figure(figsize=(3,3))
# sns.heatmap(confusion_matrix(y_test,y_test_pred_knn2), annot=True, cmap="YlGnBu",fmt='g',cbar=False)
# plt.title("Test Confusion Matrix",fontsize=16,fontweight="bold")
# plt.show()

# # Decision Tree

# In[104]:


from sklearn.tree import DecisionTreeClassifier


# In[105]:


dt = DecisionTreeClassifier(class_weight = 'balanced')
dt.fit(x_train,y_train)
y_test_pred_dt=dt.predict(x_test)


# In[106]:


Accuracy=accuracy_score(y_test,y_test_pred_dt)
Precision=precision_score(y_test,y_test_pred_dt)
Recall=recall_score(y_test,y_test_pred_dt)
F1_score=f1_score(y_test,y_test_pred_dt)
y_test_pred_prob=dt.predict_proba(x_test)[:,1]
AUC=roc_auc_score(y_test,y_test_pred_prob)

print("========== Model Performance =============")
print("Test Accuracy: %0.3f%%" %(Accuracy*100.0))
print("Precision: %0.3f%%" %(Precision*100.0))
print("Recall: %0.3f%%" %(Recall*100.0))
print("F1_score: %0.3f%%" %(F1_score*100.0))
print("AUC Score: %0.3f%%" %(AUC*100.0))
print("*******************************")
plt.figure(figsize=(3,3))
sns.heatmap(confusion_matrix(y_test,y_test_pred_dt), annot=True, cmap="YlGnBu",fmt='g',cbar=False)
plt.title("Test Confusion Matrix",fontsize=16,fontweight="bold")
plt.show()


# # Random Forest

# In[107]:


from sklearn.ensemble import RandomForestClassifier


# In[108]:


rf = RandomForestClassifier(n_estimators=20,n_jobs=-1,min_samples_leaf=0.01)
rf.fit(x_train,y_train)
y_test_pred_rf=rf.predict(x_test)


# In[109]:


Accuracy=accuracy_score(y_test,y_test_pred_rf)
Precision=precision_score(y_test,y_test_pred_rf)
Recall=recall_score(y_test,y_test_pred_rf)
F1_score=f1_score(y_test,y_test_pred_rf)
y_test_pred_prob=rf.predict_proba(x_test)[:,1]
AUC=roc_auc_score(y_test,y_test_pred_prob)

print("========== Model Performance =============")
print("Test Accuracy: %0.3f%%" %(Accuracy*100.0))
print("Precision: %0.3f%%" %(Precision*100.0))
print("Recall: %0.3f%%" %(Recall*100.0))
print("F1_score: %0.3f%%" %(F1_score*100.0))
print("AUC Score: %0.3f%%" %(AUC*100.0))
print("*******************************")
plt.figure(figsize=(3,3))
sns.heatmap(confusion_matrix(y_test,y_test_pred_rf), annot=True, cmap="YlGnBu",fmt='g',cbar=False)
plt.title("Test Confusion Matrix",fontsize=16,fontweight="bold")
plt.show()


# Importance Feature Selection

# In[110]:


from sklearn.feature_selection import RFE,SelectFromModel


# In[111]:


#Get the importances of the resulting features
importances =rf.feature_importances_

#Create a data frame for visualisation
df_feat=pd.DataFrame({"Features":x_train.columns,"Importances":importances})

df_feat.set_index('Importances')

#sort in ascending order 
df_feat=df_feat.sort_values('Importances',ascending=False)

#plot
plt.figure(figsize=(25,5))
plt.xticks(rotation=45)
sns.barplot(x='Features',y='Importances',data=df_feat)


# In[114]:


#Use LFE to eliminate the less important features
sel_rfe_tree=RFE(estimator=rf,n_features_to_select=10,step=1)
x_train_rfe_tree=sel_rfe_tree.fit_transform(x_train,y_train)
print(sel_rfe_tree.get_support())
print(sel_rfe_tree.ranking_)

y_pred_rf_lfe=sel_rfe_tree.predict(x_test)


# In[115]:


Accuracy=accuracy_score(y_test,y_pred_rf_lfe)
Precision=precision_score(y_test,y_pred_rf_lfe)
Recall=recall_score(y_test,y_pred_rf_lfe)
F1_score=f1_score(y_test,y_pred_rf_lfe)
y_test_pred_prob=rf.predict_proba(x_test)[:,1]
AUC=roc_auc_score(y_test,y_test_pred_prob)

print("========== Model Performance =============")
print("Test Accuracy: %0.3f%%" %(Accuracy*100.0))
print("Precision: %0.3f%%" %(Precision*100.0))
print("Recall: %0.3f%%" %(Recall*100.0))
print("F1_score: %0.3f%%" %(F1_score*100.0))
print("AUC Score: %0.3f%%" %(AUC*100.0))
print("*******************************")
plt.figure(figsize=(3,3))
sns.heatmap(confusion_matrix(y_test,y_pred_rf_lfe), annot=True, cmap="YlGnBu",fmt='g',cbar=False)
plt.title("Test Confusion Matrix",fontsize=16,fontweight="bold")
plt.show()


# # SVM

# In[116]:


from sklearn.svm import SVC


# In[117]:


sv=SVC()
sv.fit(x_train,y_train)
y_test_pred_svm=sv.predict(x_test)


# In[118]:


# Accuracy=accuracy_score(y_test,y_test_pred_svm)
Precision=precision_score(y_test,y_test_pred_svm)
Recall=recall_score(y_test,y_test_pred_svm)
F1_score=f1_score(y_test,y_test_pred_svm)

print("========== Model Performance =============")
print("Test Accuracy: %0.3f%%" %(Accuracy*100.0))
print("Precision: %0.3f%%" %(Precision*100.0))
print("Recall: %0.3f%%" %(Recall*100.0))
print("F1_score: %0.3f%%" %(F1_score*100.0))
print("*******************************")
plt.figure(figsize=(3,3))
sns.heatmap(confusion_matrix(y_test,y_test_pred_svm), annot=True, cmap="YlGnBu",fmt='g',cbar=False)
plt.title("Test Confusion Matrix",fontsize=16,fontweight="bold")
plt.show()


# # Boosting

# # AdaBoost

# In[119]:


from sklearn.ensemble import AdaBoostClassifier


# In[120]:


ADB = AdaBoostClassifier(base_estimator=dt,n_estimators=4)
ADB.fit(x_train,y_train)
y_test_pred_ADB=ADB.predict(x_test)


# In[121]:


Accuracy=accuracy_score(y_test,y_test_pred_ADB)
Precision=precision_score(y_test,y_test_pred_ADB)
Recall=recall_score(y_test,y_test_pred_ADB)
F1_score=f1_score(y_test,y_test_pred_ADB)
y_test_pred_prob=ADB.predict_proba(x_test)[:,1]
AUC=roc_auc_score(y_test,y_test_pred_ADB)

print("========== Model Performance =============")
print("Test Accuracy: %0.3f%%" %(Accuracy*100.0))
print("Precision: %0.3f%%" %(Precision*100.0))
print("Recall: %0.3f%%" %(Recall*100.0))
print("F1_score: %0.3f%%" %(F1_score*100.0))
print("AUC Score: %0.3f%%" %(AUC*100.0))
print("*******************************")
plt.figure(figsize=(3,3))
sns.heatmap(confusion_matrix(y_test,y_test_pred_ADB), annot=True, cmap="YlGnBu",fmt='g',cbar=False)
plt.title("Test Confusion Matrix",fontsize=16,fontweight="bold")
plt.show()


# # Gredient Boost

# In[122]:


from sklearn.ensemble import GradientBoostingClassifier


# In[123]:


GB = GradientBoostingClassifier()
GB.fit(x_train,y_train)
y_test_pred_GB=GB.predict(x_test)


# In[124]:


Accuracy=accuracy_score(y_test,y_test_pred_GB)
Precision=precision_score(y_test,y_test_pred_GB)
Recall=recall_score(y_test,y_test_pred_GB)
F1_score=f1_score(y_test,y_test_pred_GB)
y_test_pred_prob=GB.predict_proba(x_test)[:,1]
AUC=roc_auc_score(y_test,y_test_pred_GB)

print("========== Model Performance =============")
print("Test Accuracy: %0.3f%%" %(Accuracy*100.0))
print("Precision: %0.3f%%" %(Precision*100.0))
print("Recall: %0.3f%%" %(Recall*100.0))
print("F1_score: %0.3f%%" %(F1_score*100.0))
print("AUC Score: %0.3f%%" %(AUC*100.0))
print("*******************************")
plt.figure(figsize=(3,3))
sns.heatmap(confusion_matrix(y_test,y_test_pred_GB), annot=True, cmap="YlGnBu",fmt='g',cbar=False)
plt.title("Test Confusion Matrix",fontsize=16,fontweight="bold")
plt.show()


# # XgBoost

# In[126]:


from xgboost import XGBClassifier


# In[127]:


XGB = XGBClassifier()
XGB.fit(x_train,y_train)
y_test_pred_XGB=XGB.predict(x_test)


# In[128]:


Accuracy=accuracy_score(y_test,y_test_pred_XGB)
Precision=precision_score(y_test,y_test_pred_XGB)
Recall=recall_score(y_test,y_test_pred_XGB)
F1_score=f1_score(y_test,y_test_pred_XGB)
y_test_pred_prob=XGB.predict_proba(x_test)[:,1]
AUC=roc_auc_score(y_test,y_test_pred_XGB)

print("========== Model Performance =============")
print("Test Accuracy: %0.3f%%" %(Accuracy*100.0))
print("Precision: %0.3f%%" %(Precision*100.0))
print("Recall: %0.3f%%" %(Recall*100.0))
print("F1_score: %0.3f%%" %(F1_score*100.0))
print("AUC Score: %0.3f%%" %(AUC*100.0))
print("*******************************")
plt.figure(figsize=(3,3))
sns.heatmap(confusion_matrix(y_test,y_test_pred_XGB), annot=True, cmap="YlGnBu",fmt='g',cbar=False)
plt.title("Test Confusion Matrix",fontsize=16,fontweight="bold")
plt.show()


# # CatBoost

# In[130]:


from catboost import CatBoostClassifier


# In[131]:


CB = CatBoostClassifier(depth=3,border_count=300)
CB.fit(x_train,y_train)
y_test_pred_CB=CB.predict(x_test)


# In[132]:


Accuracy=accuracy_score(y_test,y_test_pred_CB)
Precision=precision_score(y_test,y_test_pred_CB)
Recall=recall_score(y_test,y_test_pred_CB)
F1_score=f1_score(y_test,y_test_pred_CB)
y_test_pred_prob=CB.predict_proba(x_test)[:,1]
AUC=roc_auc_score(y_test,y_test_pred_CB)

print("========== Model Performance =============")
print("Test Accuracy: %0.3f%%" %(Accuracy*100.0))
print("Precision: %0.3f%%" %(Precision*100.0))
print("Recall: %0.3f%%" %(Recall*100.0))
print("F1_score: %0.3f%%" %(F1_score*100.0))
print("AUC Score: %0.3f%%" %(AUC*100.0))
print("*******************************")
plt.figure(figsize=(3,3))
sns.heatmap(confusion_matrix(y_test,y_test_pred_CB), annot=True, cmap="YlGnBu",fmt='g',cbar=False)
plt.title("Test Confusion Matrix",fontsize=16,fontweight="bold")
plt.show()


# Taking Categorical variables

# In[133]:


df1_c=df1.drop(columns=['rate_of_interest','Status'])


# In[134]:


df1_c.dtypes


# In[135]:


df1_c.head()


# In[136]:


df1_c.shape


# In[137]:


x_train_1,x_test_1,y_train_1,y_test_1=train_test_split(df1_c,Target_var,test_size=0.3,random_state=0,stratify=Target_var)


# In[138]:


cat_fe=['Gender','approv_in_adv','Credit_Worthiness','open_credit','business_or_commercial','Neg_ammortization','interest_only','lump_sum_payment','Secured_by','total_units','age','Region']


# In[139]:


CB_N = CatBoostClassifier(depth=3,border_count=200,learning_rate=0.1,eval_metric='Accuracy')
CB_N.fit(x_train_1,y_train_1,cat_features=cat_fe,plot=True,eval_set=(x_test_1,y_test_1))
y_test_pred_CB_N=CB_N.predict(x_test_1)


# In[140]:


Accuracy=accuracy_score(y_test_1,y_test_pred_CB_N)
Precision=precision_score(y_test_1,y_test_pred_CB_N)
Recall=recall_score(y_test_1,y_test_pred_CB_N)
F1_score=f1_score(y_test_1,y_test_pred_CB_N)
y_test_pred_prob=CB_N.predict_proba(x_test_1)[:,1]
AUC=roc_auc_score(y_test_1,y_test_pred_CB_N)

print("========== Model Performance =============")
print("Test Accuracy: %0.3f%%" %(Accuracy*100.0))
print("Precision: %0.3f%%" %(Precision*100.0))
print("Recall: %0.3f%%" %(Recall*100.0))
print("F1_score: %0.3f%%" %(F1_score*100.0))
print("AUC Score: %0.3f%%" %(AUC*100.0))
print("*******************************")
plt.figure(figsize=(3,3))
sns.heatmap(confusion_matrix(y_test_1,y_test_pred_CB_N), annot=True, cmap="YlGnBu",fmt='g',cbar=False)
plt.title("Test Confusion Matrix",fontsize=16,fontweight="bold")
plt.show()


# # Saving the trained model

# In[141]:


import pickle


# In[142]:


filename='trained_model.sav'
pickle.dump(CB_N,open(filename,'wb'))


# Loading the saved model

# In[143]:


loaded_model=pickle.load(open('trained_model.sav','rb'))


# In[144]:


input_data=('joint','nopre','l1','nopc','b/c',10000,3000,360,'not_neg','not_int','lpsm',10000000,'home','1U',20000,500,'55-64',70,'North',37)

#changing the input data to numpy array
input_data_as_numpy_array=np.asarray(input_data)

#reshaping the data
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction=loaded_model.predict(input_data_reshaped)

print(prediction)

if(prediction[0]==0):
    print('It is less likely to become default loan.Loan is approved')
else:
    print('It is likely to become default loan.Loan is not approved')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




