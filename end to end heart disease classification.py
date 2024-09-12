#!/usr/bin/env python
# coding: utf-8

# # Predicting heart disease using machine learning
# This notebook looks into using various Python-based machine learning and data science libraries in an attempt to build a machine learning model capable of predicting whether or not someone has heart disease based on their medical attributes.
# 
# We're going to take the following approach:
# 1. Problem definition
# 2. Data
# 3. Evaluation
# 4. Features
# 5. Modelling
# 6. Experimentation
# 
# ## 1. Problem Definition
# In a statement,
# > Given clinical parameters about a patient, can we predict whether or not they have heart
# disease?
# ## 2. Data
# The original data came from the Cleavland data from the UCI Machine Learning Repository.
# https://archive.ics.uci.edu/ml/datasets/heart+Disease
# There is also a version of it available on Kaggle. https://www.kaggle.com/ronitf/heart-
# disease-uci
# ## 3. Evaluation
# > If we can reach 95% accuracy at predicting whether or not a patient has heart disease
# during the proof of concept, we'll pursue the project.
# ## 4. Features
# 
# This is where you'll get different information about each of the features in your data. You can do this via doing your own
# research (such as looking at the links above) or by talking to a subject matter expert (someone who knows about the dataset).
# Create data dictionary
# 1. age-age in years
# 2. sex (1 = male; 0 = female)
# 3. cp - chest pain type
#     * 0: Typical angina: chest pain related decrease blood supply to the heart
#     * 1: Atypical angina: chest pain not related to heart
#     * 2: Non-anginal pain: typically esophageal spasms (non heart related)
#     * 3: Asymptomatic: chest pain not showing signs of disease
# 4. trestbps - resting blood pressure (in mm Hg on admission to the hospital) anything above 130-140 is typically cause for
# concern
# 5. chol - serum cholestoral in mg/dl
#     * serum LDL + HDL +.2* triglycerides
#     * above 200 is cause for concern
# 6. fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
#     * '>126' mg/dL signals diabetes
# 7. resteca - resting electrocardiographic results
#     * 0: Nothing to note
#     * 1: ST-T Wave phnormality
#         * can range from mild symptoms to severe problems
#         * signals non-normal heart beat
#     * 2: Possible or definite left ventricular hypertrophy
#         * Enlarged heart's main pumping chamber
# 8. thalach - maximum heart rate achieved
# 9. exang - exercise induced angina (1 = yes; 0 = no)
# 10. oldpeak - ST depression induced by exercise relative to rest looks at stress of hea
# stress more
# 11. slope the slope of the peak exercise ST segment
#     * 0: Upsloping: better heart rate with excercise (uncommon)
#      * 1: Flatsloping: minimal change (typical healthy heart)
#     * 2: Downslopins: signs of unhealthy heart
# 12. ca number of major vessels (0-3) colored by flourosopy
#      * colored vessel means the doctor can see the blood passing through
#      * the more blood movement the better (no clots)
# 13. thal - thalium stress result
#     * 1,3: normal
#     * 6: fixed defect: used to be defect but ok now
#     * 7: reversable defect: no proper blood movement when excercising
# 14. target - have disease or not (1=yes, 0=no) (= the predicted attribute)

# ## Preparing the tools
# We're going to use pandas ,matplotlib and numpy for data analysis and manipulation

# In[1]:


# Importing all the tools we need

#Regular EDA and plotting libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# we want plots inside our jupyter notebook
get_ipython().run_line_magic('matplotlib', 'inline')
#model from sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#model evaluation
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import RocCurveDisplay


# # Load Data

# In[2]:


df=pd.read_csv('C:\\Users\\Akanksha\\OneDrive\\Desktop\\heart_disease_data.csv')
df


# ## Data Exploration (EDA)
# find more about data
# 1. whaat questions are we trying to solve?
# 2. what kind of data we have and we treat different types?
# 3. what's missing from data and how do you deal with it?
# 4. Where are the outliers and why should you care about them
# 5. How can you add, change and remove features to get more information?

# In[3]:


df.head()


# In[4]:


# hoc many of each class
df['target'].value_counts()


# In[5]:


df['target'].value_counts().plot(kind='bar',color=['salmon','lightblue']);


# In[6]:


#different info
df.info()


# In[7]:


# missing values?
df.isna().sum()


# In[8]:


df.describe()


# ## Heart Disease Frequency according to sex
# 

# In[9]:


df.sex.value_counts()


# In[10]:


#compare target column with sex column
pd.crosstab(df.target,df.sex)
''' given cross tab shows that 75 % women can have heartdisease and 45% men can have heart diseas '''


# In[11]:


#Create a plot of crosstab
pd.crosstab(df.target,df.sex).plot(kind='bar',
                                   figsize=(10,6),
                                   color=['salmon','lightblue']
                                  )
plt.title('Heart disease frequency for sex')
plt.xlabel('0=No disease, 1= Disease')
plt.ylabel('Amount')
plt.legend(['Female','Male'])
plt.xticks(rotation=0);


# In[12]:


df['thalach'].value_counts()


# ## Age vs Max Heart Rate for Heart Disease

# In[13]:


#create figure
plt.figure(figsize=(10,6))
plt.scatter(df.age[df.target==1],
           df.thalach[df.target==1],
           c='salmon')
plt.scatter(df.age[df.target==0],
           df.thalach[df.target==0],
           c='lightblue')
plt.title('Age V/S max Heart Rate')
plt.xlabel("Age")
plt.ylabel("Max heart rate")
plt.legend(['Disease','No disease']);


# In[14]:


#check the distribution of age column
df.age.plot.hist();


# ### Heart Disease frquency vs chestpain type
# 3. cp chest pain type
#     * 0: Typical angina: chest pain related decrease blood supply to the heart
#     * 1: Atypical angina: chest pain not related to heart
#     * 2: Non-anginal pain: typically esophageal spasms (non heart related)
#     * 3: Asymptomatic: chest pain not showing signs of disease

# In[15]:


pd.crosstab(df.cp,df.target)


# In[16]:


#make cross tab more visual
pd.crosstab(df.cp,df.target).plot(kind='bar',
                                 figsize=(10,6),
                                 color=['salmon','lightblue'])
plt.title('Heart Disease Frequency per chest pain type')
plt.xlabel('Chest pain Type')
plt.ylabel('Amount')
plt.legend(['No Disease','Disease'])
plt.xticks(rotation=0);


# In[17]:


df.head()


# In[18]:


# make correlation matrix
df.corr()


# In[19]:


#correlation visualization
corr_matrix=df.corr()
fig,ax=plt.subplots(figsize=(15,10))
ax=sns.heatmap(corr_matrix,
              annot=True,
              linewidths=0.5,
              fmt='.2f',
              cmap='viridis')
# bottom,top=ax.get_ylim()
# ax.set_ylim(bottom+0.5,top-0.5)


# # 5. Modelling

# In[20]:


df.head()


# In[21]:


#split data into x and y
x=df.drop('target',axis=1)
y=df['target']


# In[22]:


#spliting data into train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# Now we've got our data split into training and test sets, it's time to build a training model
# 
# We'll train it (find the patterns) on the training set.
# 
# And we'll test it (use the patterns) on the test set.
# 
# We're going to try 3 different machine learning models:
# 1. Logistic Regression
# 2. K-Nearest Neighbours Classifier
# 3. Random Forest Classifier

# In[23]:


#puts models in dictionary
models={'Logistic Regression':LogisticRegression(),
       "KNN":KNeighborsClassifier(),
       "Random Forest":RandomForestClassifier()}
def fit_and_score(models,x_train,x_test,y_train,y_test):
    '''Fits and evaluates given machine learning models.
        models a dict of differetn Scikit-Learn machine learning models
        x_train training data (no labels)
        xtest testing data (no labels)
        y_train training labels
        y_test test labels
    '''
    #set random seed
    np.random.seed(42)
    #make a dictionary to keep model scores.
    model_scores={}
    for name ,model in models.items():
        model.fit(x_train,y_train)
        #evaluate model and append the score
        model_scores[name]=model.score(x_test,y_test)
    return model_scores


# In[24]:


model_scores=fit_and_score(models,
                           x_train,
                           x_test,
                           y_train,
                           y_test,
                           )
model_scores


# ## Model Comparision

# In[25]:


model_compare=pd.DataFrame(model_scores,index=['accuracy'])
model_compare.T.plot.bar();


# Now we've got a baseline model... and we know a model's first
# predictions aren't always what we should based our next steps off.
# What should do?
# Let's look at the following:
# * Hypyterparameter tuning
# * Feature importance
# * Confusion matrix
# * Precision
# * Recall
# * F1 score
# * Classification report
# * Roc curve
# * Area under the curve
# 
# ### Hyperparameter tuning

# In[26]:


#Lets tune KNN
train_scores=[]
test_scores=[]
neighbors=range(1,21)
knn=KNeighborsClassifier()
#loop through different n_neighbors
for i in neighbors:
    knn.set_params(n_neighbors=i)
    #fit the algorithm
    knn.fit(x_train,y_train)
    #update training scores list
    train_scores.append(knn.score(x_train,y_train))
    #update test scores list
    test_scores.append(knn.score(x_test,y_test))
train_scores    


# In[27]:


plt.plot(neighbors,train_scores,label="Train score")
plt.plot(neighbors,test_scores,label='Test score')
plt.xticks(np.arange(1,21,1))
plt.xlabel('No of neighbors')
plt.ylabel('Model score')
plt.legend()
print(f"Maximum KNN score on the test data:{max(test_scores)*100:.2f}%")


# ### Hyperparameter tuning with RandomizedSearchCV

# We're going to tune our:
# * logistic regression model 
# * RandomForestClassifer

# In[28]:


# Create a hyperparameter grid for logistics regression
log_reg_grid={'C':np.logspace(-4,4,20),
             "solver":['liblinear']}
#create hyperparameter grid for randomforestclassifier
rf_grid={"n_estimators":np.arange(10,1000,50),
        "max_depth":[None,3,5,10],
        "min_samples_split":np.arange(2,20,2),
        'min_samples_leaf':np.arange(1,20,2)}


# Now we've got hyperparameters grids setup for each of our model,lets tune them using randomizedseachcv

# In[29]:


#Tune logistic Regression
np.random.seed(42)
#setup hyperparameter search for logistics regression
rs_log_reg=RandomizedSearchCV(LogisticRegression(),
                              param_distributions=log_reg_grid,
                             cv=5,
                             n_iter=20,
                             verbose=True)
rs_log_reg.fit(x_train,y_train)


# In[30]:


rs_log_reg.best_params_


# In[31]:


rs_log_reg.score(x_test,y_test)


# In[32]:


# let tune randomforest classifier
np.random.seed(42)
rs_rf=RandomizedSearchCV(RandomForestClassifier(),
                        param_distributions=rf_grid,
                        cv=5,
                        n_iter=20,
                         verbose=True
                        )
rs_rf.fit(x_train,y_train)


# In[33]:


#find best parameters
rs_rf.fit(x_train,y_train)


# ### hyperparameter tuning using gridsearchcv

# In[34]:


#different hyperparameters for logistics regression
log_reg_grid={'C':np.logspace(-4,4,30),
             'solver':['liblinear']}
gs_log_reg=GridSearchCV(LogisticRegression(),
                       param_grid=log_reg_grid,
                       cv=5,
                       verbose=True)
gs_log_reg.fit(x_train,y_train)


# In[35]:


#evaluate gridsearchcv 
gs_log_reg.score(x_test,y_test)


# Evaluting our tuned machine learning classifier, beyond
# accuracy
# * ROC curve and AUC score
# * Confusion matrix
# * Classification report
# * Precision
# * Recall
# * F1-score
# ... and i would be great if cross-validation was used where possible.

# In[36]:


#make predictions with tuned model
y_preds=gs_log_reg.predict(x_test)
y_preds


# In[40]:


# plot ROC Curve and calculate 
RocCurveDisplay.from_predictions(y_test,y_preds)


# In[42]:


#confusion matrix
print(confusion_matrix(y_test,y_preds))


# In[47]:


sns.set_theme(font_scale=1.5)
def plot_conf_mat(y_test,y_preds):
    '''plots confusion matrix'''
    fig,ax=plt.subplots(figsize=(3,3))
    ax=sns.heatmap(confusion_matrix(y_test,y_preds),
                   annot=True,
                   cbar=False)
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
plot_conf_mat(y_test,y_preds)
            


# In[51]:


#classification report
print(classification_report(y_test,y_preds))


# ### Calculate evaluation metrics using cross validation

# In[54]:


# check best hyperparameters
gs_log_reg.best_params_


# In[56]:


# create a new classifier with best parameters
clf=LogisticRegression(C=0.20433597178569418,
                       solver='liblinear')


# In[61]:


# Cross validated accuracy
cv_acc=cross_val_score(clf,
                      x,
                      y,
                      cv=5,scoring='accuracy')
cv_acc


# In[62]:


cv_acc=np.mean(cv_acc)
cv_acc


# In[63]:


# Cross validated precison
cv_precision=cross_val_score(clf,
                            x,
                            y,
                            cv=5,
                            scoring='precision'
                            )
cv_precision=np.mean(cv_precision)
cv_precision


# In[66]:


# Cross validated f1_score
cv_f1=cross_val_score(clf,
                            x,
                            y,
                            cv=5,
                            scoring='f1'
                            )
cv_f1=np.mean(cv_f1)
cv_f1


# In[67]:


# Cross validated recall
cv_recall=cross_val_score(clf,
                            x,
                            y,
                            cv=5,
                            scoring='recall'
                            )
cv_recall=np.mean(cv_recall)
cv_recall


# In[72]:


# vizualize our cross validated matrix
cv_metrics=pd.DataFrame({"Accuracy":cv_acc,
                       "Precision":cv_precision,
                       "Recall":cv_recall,
                       "F1":cv_f1},
                       index=[0])
cv_metrics.T.plot.bar(title='Cross validated classification report');


# ## Feature Importance
# Feature importance is another as aksing, 'which features contributed most to the outcomes of the model and how did the contribute?'
# 
# Finding feautre importance is different for each machine learning model. ( search for (MODEL NAME) feature importance
# 
# Let's find feature importance for logistic regression model.

# In[76]:


# Fit an instance of logisticsRegression 
gs_log_reg.best_params_
clf=LogisticRegression(C=0.20433597178569418,
                      solver='liblinear')
clf.fit(x_train,y_train)


# In[77]:


# check coef_(reation between each feature and target )
clf.coef_


# In[82]:


feature_dict=dict(zip(df.columns,list(clf.coef_[0])))
feature_dict


# In[88]:


# Vizualize feature importance
feature_df=pd.DataFrame(feature_dict,index=[0])
feature_df.T.plot.bar(title='Feature Importance',legend=False)


# In[ ]:




