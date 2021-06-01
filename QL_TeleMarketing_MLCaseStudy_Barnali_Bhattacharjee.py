#!/usr/bin/env python
# coding: utf-8

# ## Import the necessary libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing


# In[2]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 5, 4
sns.set_style('whitegrid')


# ## Import the raw dataset

# In[4]:


address = 'C:/Users/Burns/Desktop/QL Case Study Analysis/bank-additional/bank-additional/Bank_Raw_Data_Full.csv'
raw_df = pd.read_csv(address)
raw_df.head()


# In[5]:


raw_df.info()


# ## Rename columns

# In[6]:


raw_df.rename(columns={'emp.var.rate': 'emp_var_rate'}, inplace=True)
raw_df.rename(columns={'cons.price.idx': 'cons_price_idx'}, inplace=True)
raw_df.rename(columns={'cons.conf.idx': 'cons_conf_idx'}, inplace=True)
raw_df.rename(columns={'nr.employed': 'nr_employed'}, inplace=True)


# In[7]:


raw_df.columns


# In[8]:


#dropping duration because we will not have this field until we know the outcome
raw_df = raw_df.drop(['duration'], axis = 1)


# ## Understanding of the data by each variable

# In[9]:


print(raw_df['age'].plot.hist())


# In[10]:


age_p95 = raw_df['age'].quantile(0.95)
print(age_p95)
raw_df.age.loc[raw_df.age > age_p95]  = age_p95
print(raw_df['age'].plot.hist())


# In[11]:


print(raw_df['campaign'].value_counts())
raw_df['campaign'].value_counts().plot.bar()


# In[12]:


campaign_p95 = raw_df['campaign'].quantile(0.95)
print(campaign_p95)
raw_df.campaign.loc[raw_df.campaign > campaign_p95]  = campaign_p95
print(raw_df['campaign'].plot.hist())


# In[13]:


print(raw_df['cons_price_idx'].plot.hist())


# In[14]:


print(raw_df['cons_conf_idx'].plot.hist())


# In[15]:


print(raw_df['euribor3m'].plot.hist())


# In[16]:


sns.set_theme(style="whitegrid")


# In[17]:


ax = sns.boxplot(x="y", y="euribor3m", data=raw_df)


# In[18]:


print(raw_df['nr_employed'].plot.hist())


# In[19]:


ax = sns.boxplot(x="y", y="age", data=raw_df)
# age distribution is similar for both the groups


# In[20]:


ax = sns.boxplot(x="y", y="campaign", data=raw_df)


# In[21]:


print(raw_df['job'].value_counts())
raw_df['job'].value_counts().plot.bar()


# In[22]:


print(raw_df['marital'].value_counts())
raw_df['marital'].value_counts().plot.bar()


# In[23]:


print(raw_df['education'].value_counts())
raw_df['education'].value_counts().plot.bar()


# In[24]:


print(raw_df['default'].value_counts())
raw_df['default'].value_counts().plot.bar()


# In[25]:


print(raw_df['housing'].value_counts())
raw_df['housing'].value_counts().plot.bar()


# In[26]:


print(raw_df['loan'].value_counts())
raw_df['loan'].value_counts().plot.bar()


# In[27]:


print(raw_df['pdays'].value_counts())
raw_df['pdays'].value_counts().plot.bar()


# In[28]:


print(raw_df['previous'].value_counts())
raw_df['previous'].value_counts().plot.bar()


# In[29]:


print(raw_df['poutcome'].value_counts())
raw_df['poutcome'].value_counts().plot.bar()


# ## Feature engineering

# In[30]:


## Reduce number of categories and impute 'unknown' by mode of the filed which is employed
raw_df.loc[raw_df['job'] == 'housemaid', 'job_segment'] = 'employed'
raw_df.loc[raw_df['job'] == 'services', 'job_segment'] = 'employed'
raw_df.loc[raw_df['job'] == 'admin.', 'job_segment'] = 'employed'
raw_df.loc[raw_df['job'] == 'blue-collar', 'job_segment'] = 'employed'
raw_df.loc[raw_df['job'] == 'technician', 'job_segment'] = 'employed'
raw_df.loc[raw_df['job'] == 'management', 'job_segment'] = 'employed'
raw_df.loc[raw_df['job'] == 'self-employed', 'job_segment'] = 'employed'
raw_df.loc[raw_df['job'] == 'entrepreneur', 'job_segment'] = 'employed'
raw_df.loc[raw_df['job'] == 'retired', 'job_segment'] = 'retired'
raw_df.loc[raw_df['job'] == 'unemployed', 'job_segment'] = 'unemployed'
raw_df.loc[raw_df['job'] == 'unknown', 'job_segment'] = 'employed'
raw_df.loc[raw_df['job'] == 'student', 'job_segment'] = 'student'


# In[31]:


## Impute 'unknown' by mode of the filed which is married
raw_df.loc[raw_df['marital'] == 'unknown', 'marital'] = 'married'


# In[32]:


print(raw_df['marital'].value_counts())
raw_df['marital'].value_counts().plot.bar()


# In[33]:


## Reduce number of categories
raw_df.loc[raw_df['education'] == 'basic.4y', 'edu_segment'] = 'basic'
raw_df.loc[raw_df['education'] == 'basic.6y', 'edu_segment'] = 'basic'
raw_df.loc[raw_df['education'] == 'basic.9y', 'edu_segment'] = 'basic'
raw_df.loc[raw_df['education'] == 'high.school', 'edu_segment'] = 'higher_studies'
raw_df.loc[raw_df['education'] == 'professional.course', 'edu_segment'] = 'higher_studies'
raw_df.loc[raw_df['education'] == 'university.degree', 'edu_segment'] = 'higher_studies'
raw_df.loc[raw_df['education'] == 'unknown', 'edu_segment'] = 'unknown'
raw_df.loc[raw_df['education'] == 'illiterate', 'edu_segment'] = 'illiterate'


# In[34]:


def func(row):
    if row['housing'] == 'yes' or row['loan'] == 'yes':
        return 'yes'
    elif row['housing'] == 'unknown' and row['loan'] == 'unknown':
        return 'unknown'
    else:
        return 'no'

raw_df['housing_loan'] = raw_df.apply(func, axis=1)


# In[35]:


def func(row):
    if row['month'] == 'jan' or row['month'] == 'feb' or row['month'] == 'mar' or row['month'] == 'apr' or row['month'] == 'sep' or row['month'] == 'oct'or row['month'] == 'nov' or row['month'] == 'dec' :
        return 'fall_spring'
    else:
        return 'summer'

raw_df['month_seg'] = raw_df.apply(func, axis=1)


# In[36]:


def func(row):
    if row['day_of_week'] == 'tue' or row['day_of_week'] == 'wed' or row['day_of_week'] == 'thu':
        return 'tue_thu'
    else:
        return 'mon_fri'

raw_df['day_of_week_seg'] = raw_df.apply(func, axis=1)


# In[37]:


raw_df['previous_flag'] = np.where(raw_df['previous'] > 0, 1, 0 )
raw_df['previous_flag'] = raw_df['previous_flag'].astype(str)


# In[38]:


print(raw_df['previous_flag'].value_counts())
raw_df['previous_flag'].value_counts().plot.bar()


# ## Check the assumptions of Logistics Regression

# #### 1. Target variable is categorical
# #### 2. Checking for missing values
# #### 3.Independence between features
# #### 4. Data size is large

# In[39]:


sns.countplot(x='y', data=raw_df, palette='hls')


# #### The target variable is binary and there is a class imbalance problem

# In[40]:


raw_df.isnull().sum()


# ## Create dummy variables

# #### job

# In[41]:


label_encoder = LabelEncoder()
job_cat = raw_df['job_segment']
job_encoded = label_encoder.fit_transform(job_cat)
pd.set_option('display.max.rows', None)
job_encoded[0:216]


# In[42]:


#0-employed/1-retired/2-student/3-unemployed
binary_encoder = OneHotEncoder(categories = 'auto')
job_1hot = binary_encoder.fit_transform(job_encoded.reshape(-1,1))
job_1hot_mat = job_1hot.toarray()
job_DF = pd.DataFrame(job_1hot_mat, columns = ['employed','retired','student','unemployed'])
job_DF.head()


# #### marital

# In[43]:


label_encoder = LabelEncoder()
marital_cat = raw_df['marital']
marital_encoded = label_encoder.fit_transform(marital_cat)
pd.set_option('display.max.rows', None)
# marital_encoded[0:1000]


# In[44]:


#0-divorced/1-married/2-single
binary_encoder = OneHotEncoder(categories = 'auto')
marital_1hot = binary_encoder.fit_transform(marital_encoded.reshape(-1,1))
marital_1hot_mat = marital_1hot.toarray()
marital_DF = pd.DataFrame(marital_1hot_mat, columns = ['divorced','married','single'])
# marital_DF.head()


# #### education

# In[45]:


label_encoder = LabelEncoder()
education_cat = raw_df['edu_segment']
education_encoded = label_encoder.fit_transform(education_cat)
pd.set_option('display.max.rows', None)
education_encoded[0:216]


# In[46]:


#0-basic/1-higher_studies/2-illiterate/3-unknown
binary_encoder = OneHotEncoder(categories = 'auto')
education_1hot = binary_encoder.fit_transform(education_encoded.reshape(-1,1))
education_1hot_mat = education_1hot.toarray()
education_DF = pd.DataFrame(education_1hot_mat, columns = ['basic','higher_studies','illiterate','edu_unknown'])
education_DF.head()


# #### default

# In[47]:


label_encoder = LabelEncoder()
default_cat = raw_df['default']
default_encoded = label_encoder.fit_transform(default_cat)
pd.set_option('display.max.rows', None)
default_encoded[0:216]


# In[48]:


#0-no/1-unknown/2-yes
binary_encoder = OneHotEncoder(categories = 'auto')
default_1hot = binary_encoder.fit_transform(default_encoded.reshape(-1,1))
default_1hot_mat = default_1hot.toarray()
default_DF = pd.DataFrame(default_1hot_mat, columns = ['default_no','default_unknown','default_yes'])
default_DF.head()


# #### housing

# In[49]:


# raw_df['housing'].unique()


# In[50]:


# label_encoder = LabelEncoder()
# housing_cat = raw_df['housing']
# housing_encoded = label_encoder.fit_transform(housing_cat)
# pd.set_option('display.max.rows', None)
# housing_encoded[0:216]


# In[51]:


#0-no/1-unknown/2-yes
# binary_encoder = OneHotEncoder(categories = 'auto')
# housing_1hot = binary_encoder.fit_transform(housing_encoded.reshape(-1,1))
# housing_1hot_mat = housing_1hot.toarray()
# housing_DF = pd.DataFrame(housing_1hot_mat, columns = ['housing_no','housing_unknown','housing_yes'])
# housing_DF.head()


# #### loan

# In[52]:


# raw_df['loan'].unique()


# In[53]:


# label_encoder = LabelEncoder()
# loan_cat = raw_df['loan']
# loan_encoded = label_encoder.fit_transform(loan_cat)
# pd.set_option('display.max.rows', None)
# loan_encoded[0:216]


# In[54]:


#0-no/1-unknown/2-yes
# binary_encoder = OneHotEncoder(categories = 'auto')
# loan_1hot = binary_encoder.fit_transform(loan_encoded.reshape(-1,1))
# loan_1hot_mat = loan_1hot.toarray()
# loan_DF = pd.DataFrame(loan_1hot_mat, columns = ['loan_no','loan_unknown','loan_yes'])
# loan_DF.head()


# In[55]:


label_encoder = LabelEncoder()
housing_loan_cat = raw_df['housing_loan']
housing_loan_encoded = label_encoder.fit_transform(housing_loan_cat)
pd.set_option('display.max.rows', None)
housing_loan_encoded[0:216]


# In[56]:


#0-no/1-unknown/2-yes
binary_encoder = OneHotEncoder(categories = 'auto')
housing_loan_1hot = binary_encoder.fit_transform(housing_loan_encoded.reshape(-1,1))
housing_loan_1hot_mat = housing_loan_1hot.toarray()
housing_loan_DF = pd.DataFrame(housing_loan_1hot_mat, columns = ['housing_loan_no','housing_loan_unknown','housing_loan_yes'])
housing_loan_DF.head()


# #### contact

# In[57]:


label_encoder = LabelEncoder()
contact_cat = raw_df['contact']
contact_encoded = label_encoder.fit_transform(contact_cat)
pd.set_option('display.max.rows', None)
contact_encoded[0:216]


# In[58]:


#0-cellular/1-telephone
binary_encoder = OneHotEncoder(categories = 'auto')
contact_1hot = binary_encoder.fit_transform(contact_encoded.reshape(-1,1))
contact_1hot_mat = contact_1hot.toarray()
contact_DF = pd.DataFrame(contact_1hot_mat, columns = ['cellular','telephone'])
contact_DF.head()


# #### month

# In[59]:


# label_encoder = LabelEncoder()
# month_cat = raw_df['month']
# month_encoded = label_encoder.fit_transform(month_cat)
# pd.set_option('display.max.rows', None)
# month_encoded[18800:20000]


# In[60]:


#0-apr/1-aug/2-dec/3-jul/4-jun/5-mar/6-may/7-nov/8-oct/9-sep
# binary_encoder = OneHotEncoder(categories = 'auto')
# month_1hot = binary_encoder.fit_transform(month_encoded.reshape(-1,1))
# month_1hot_mat = month_1hot.toarray()
# month_DF = pd.DataFrame(month_1hot_mat, columns = ['apr','aug','dec','jul','jun','mar','may','nov','oct','sep'])
# month_DF.head()


# In[61]:


label_encoder = LabelEncoder()
month_seg_cat = raw_df['month_seg']
month_seg_encoded = label_encoder.fit_transform(month_seg_cat)
pd.set_option('display.max.rows', None)
pd.set_option('display.max_rows', None)
month_seg_encoded[0:216]


# In[62]:


raw_df['month_seg'].unique()


# In[63]:


#0-fall_spring/1-summer
binary_encoder = OneHotEncoder(categories = 'auto')
month_seg_1hot = binary_encoder.fit_transform(month_seg_encoded.reshape(-1,1))
month_seg_1hot_mat = month_seg_1hot.toarray()
month_seg_DF = pd.DataFrame(month_seg_1hot_mat, columns = ['fall_spring','summer'])
month_seg_DF.head()


# #### day_of_week

# In[64]:


# label_encoder = LabelEncoder()
# day_of_week_cat = raw_df['day_of_week']
# day_of_week_encoded = label_encoder.fit_transform(day_of_week_cat)
# pd.set_option('display.max.rows', None)
# day_of_week_encoded[18800:20000]


# In[65]:


# #0-fri/1-mon/2-thu/3-tue/4-wed
# binary_encoder = OneHotEncoder(categories = 'auto')
# day_of_week_1hot = binary_encoder.fit_transform(day_of_week_encoded.reshape(-1,1))
# day_of_week_1hot_mat = day_of_week_1hot.toarray()
# day_of_week_DF = pd.DataFrame(day_of_week_1hot_mat, columns = ['fri','mon','thu','tue','wed'])
# day_of_week_DF.head()


# In[66]:


label_encoder = LabelEncoder()
day_of_week_seg_cat = raw_df['day_of_week_seg']
day_of_week_seg_encoded = label_encoder.fit_transform(day_of_week_seg_cat)
pd.set_option('display.max.rows', None)
day_of_week_seg_encoded[18800:20000]


# In[67]:


raw_df.head()


# In[68]:


#0-mon_fri/1-tue_thu
binary_encoder = OneHotEncoder(categories = 'auto')
day_of_week_seg_1hot = binary_encoder.fit_transform(day_of_week_seg_encoded.reshape(-1,1))
day_of_week_seg_1hot_mat = day_of_week_seg_1hot.toarray()
day_of_week_seg_DF = pd.DataFrame(day_of_week_seg_1hot_mat, columns = ['mon_fri','tue_thu'])
day_of_week_seg_DF.head()


# #### poutcome

# In[69]:


label_encoder = LabelEncoder()
poutcome_cat = raw_df['poutcome']
poutcome_encoded = label_encoder.fit_transform(poutcome_cat)
pd.set_option('display.max.rows', None)
poutcome_encoded[18800:20000]


# In[70]:


#0-failure/1-nonexistent/2-success
binary_encoder = OneHotEncoder(categories = 'auto')
poutcome_1hot = binary_encoder.fit_transform(poutcome_encoded.reshape(-1,1))
poutcome_1hot_mat = poutcome_1hot.toarray()
poutcome_DF = pd.DataFrame(poutcome_1hot_mat, columns = ['failure','nonexistent','success'])
poutcome_DF.head()


# #### y

# In[71]:


label_encoder = LabelEncoder()
y_cat = raw_df['y']
y_encoded = label_encoder.fit_transform(y_cat)
pd.set_option('display.max.rows', None)
y_encoded[18800:20000]


# In[72]:


#0-no/1-yes
binary_encoder = OneHotEncoder(categories = 'auto')
y_1hot = binary_encoder.fit_transform(y_encoded.reshape(-1,1))
y_1hot_mat = y_1hot.toarray()
y_DF = pd.DataFrame(y_1hot_mat, columns = ['y_no','target'])
y_DF.head()


# In[73]:


raw_df.dtypes


# In[74]:


# drop the original categorical variable and keep the dummies
raw_df.drop(['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome','y', 'job_segment','edu_segment','housing_loan', 'month_seg', 'day_of_week_seg'],axis = 1, inplace = True)
raw_df.head()


# In[75]:


raw_df_dummy = pd.concat([raw_df,job_DF,marital_DF,education_DF,default_DF,housing_loan_DF,contact_DF, month_seg_DF, day_of_week_seg_DF,poutcome_DF, y_DF], axis = 1, verify_integrity=True).astype(float).copy()
raw_df_dummy[0:5]


# ## Feature Reduction using Business Knowledge, Correlation and VIF 
# ## Detect Multicollineary and reduce features accordingly

# In[76]:


#drop one dummy variable for each categorical field because of multicillinearity. Example - if we have a categorical variable with 3 values we only need 2 dummy variables to represt that field
raw_df_dummy.drop(['unemployed','divorced','edu_unknown','default_unknown','housing_loan_unknown','cellular','nonexistent', 'summer', 'mon_fri'],axis = 1, inplace = True)
raw_df_dummy.head()


# In[77]:


#dropping y_no as we only need the target field (y = yes) as outcome variable
raw_df_dummy.drop(['y_no'],axis = 1, inplace = True)
raw_df_dummy.head()


# In[78]:


#dropping previous field because we have created the previous_flag using previous
raw_df_dummy.drop(['previous'],axis = 1, inplace = True)
raw_df_dummy.head()


# In[79]:


#dropping poutcome because 87% of cases poutcome is non existent which will be representated by previous_flag = 0, hence these fields are highly correlated
#dropping pdays for the same reason
raw_df_dummy.drop(['failure', 'success', 'pdays'],axis = 1, inplace = True)
raw_df_dummy.head()


# In[80]:


raw_df_dummy.corr()


# In[81]:


sns.heatmap(raw_df_dummy.corr())


# In[82]:


#dropping nr_emplayed as it has 95% correlation with euribor
raw_df_dummy.drop(['nr_employed'],axis = 1, inplace = True)
raw_df_dummy.head()


# In[83]:


# the independent variables set
X = raw_df_dummy.drop(['target'], axis = 1)
  
# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
  
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
  
print(vif_data)


# In[84]:


#dropping fields with higest vif

# the independent variables set
X = raw_df_dummy.drop(['target', 'cons_price_idx'], axis = 1)
  
# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
  
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
  
print(vif_data)


# In[85]:


#dropping fields with higest vif


# the independent variables set
X = raw_df_dummy.drop(['target', 'cons_price_idx', 'euribor3m'], axis = 1)
  
# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
  
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
  
print(vif_data)


# In[86]:


#dropping fields with higest vif


# the independent variables set
X = raw_df_dummy.drop(['target', 'cons_price_idx', 'euribor3m', 'cons_conf_idx'], axis = 1)
  
# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
  
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
  
print(vif_data)


# In[87]:


#dropping fields with higest vif


# the independent variables set
X = raw_df_dummy.drop(['target', 'cons_price_idx', 'euribor3m', 'cons_conf_idx', 'employed'], axis = 1)
  
# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
  
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
  
print(vif_data)


# In[88]:


#dropping fields with higest vif


# the independent variables set
X = raw_df_dummy.drop(['target', 'cons_price_idx', 'euribor3m', 'cons_conf_idx', 'employed', 'age'], axis = 1)
  
# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
  
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
  
print(vif_data)


# In[89]:


#dropping fields with higest vif


# the independent variables set
X = raw_df_dummy.drop(['target', 'cons_price_idx', 'euribor3m', 'cons_conf_idx', 'employed', 'age', 'housing_loan_yes'], axis = 1)
  
# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
  
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
  
print(vif_data)


# In[90]:


#dropping fields with higest vif


# the independent variables set
X = raw_df_dummy.drop(['target', 'cons_price_idx', 'euribor3m', 'cons_conf_idx', 'employed', 'age', 'housing_loan_yes','higher_studies'], axis = 1)
  
# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
  
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
  
print(vif_data)


# In[91]:


model_df = raw_df_dummy.drop(['cons_price_idx', 'euribor3m', 'cons_conf_idx', 'employed', 'age', 'housing_loan_yes', 'higher_studies'], axis = 1)


# In[92]:


model_df.shape


# ## Feature Reduction

# ## Model Building & Validation

# In[93]:


# Get the significant predictors
lr_model= smf.logit(formula="target~ campaign + emp_var_rate + previous_flag+ C(retired) + C(student) + C(married) + C(single)+C(basic) + C(illiterate) +C(default_no) + C(default_yes) + C(housing_loan_no) + C(telephone) + C(fall_spring) +C(tue_thu)", data = model_df).fit()
lr_model.summary()


# In[94]:


# Keep the factors with p value <= 0.05
model_df_lr = model_df.drop(['married', 'illiterate', 'default_yes', 'housing_loan_no'], axis = 1)


# #### Split the data into train and test

# In[95]:


X_train,X_test,y_train,y_test = train_test_split(model_df_lr.drop('target', axis = 1),
                                                 model_df_lr['target'], test_size = 0.2,
                                                 random_state = 9999)


# In[96]:


print(X_train.shape)


# In[97]:


print(y_train.shape)


# In[98]:


X_train[0:5]


# In[99]:


LRClassifier = LogisticRegression(solver = 'liblinear', class_weight='balanced')
LRClassifier.fit(X_train,y_train)


# In[100]:


LRClassifier.get_params()


# In[101]:


X_train.dtypes


# In[102]:


LRClassifier.coef_


# In[103]:


LRClassifier.intercept_


# In[104]:


y_lr_pred = LRClassifier.predict(X_test)


# In[105]:


print(classification_report(y_test, y_lr_pred))


# In[106]:


y_lr_test_pred = cross_val_predict(LRClassifier, X_test, y_test, cv = 9)
confusion_matrix(y_test, y_lr_test_pred)


# In[107]:


precision_score(y_test, y_lr_test_pred)


# In[108]:


recall_score(y_test, y_lr_test_pred)


# In[109]:



f1_score(y_test, y_lr_test_pred)


# In[110]:


balanced_accuracy_score(y_test, y_lr_test_pred)


# ## Random Forest

# In[111]:


RFClassifier = RandomForestClassifier(n_estimators=99, max_depth = 3, min_samples_leaf = 100, random_state=0, class_weight='balanced' )
y_train_array = np.ravel(y_train)
RFClassifier.fit(X_train, y_train_array)
y_rf_pred = RFClassifier.predict(X_test)


# In[112]:


print(metrics.classification_report(y_test, y_rf_pred))


# In[113]:


precision_score(y_test, y_rf_pred)


# In[114]:


recall_score(y_test, y_rf_pred)


# In[115]:


f1_score(y_test, y_rf_pred)


# In[116]:


balanced_accuracy_score(y_test, y_rf_pred)


# In[117]:


X_train.dtypes


# In[118]:


featureImportances = pd.Series(RFClassifier.feature_importances_).sort_values(ascending=False)
print(featureImportances)

sns.barplot(x=round(featureImportances,4), y=featureImportances)
plt.xlabel('Features Importance')
plt.show()


# In[119]:


plt.figure(figsize=(25,20))
_ = tree.plot_tree(RFClassifier.estimators_[18], feature_names=X.columns, class_names =True, filled=True, proportion=True)

