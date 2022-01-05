#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns


# In[2]:


df=pd.read_csv('/content/avila-tr.txt',sep=',',header=None)


# In[3]:


df.columns =['A', 'B', 'C', 'D','E','F','G','H','I','J','target']


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.dtypes


# In[7]:


df.isna().sum()


# In[8]:


df['target'].value_counts().plot(kind="bar")


# In[9]:


val = df.target.value_counts()


# In[10]:


corr = df.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)


# In[11]:


df.hist(bins=50, figsize=(20,15))
plt.show();


# In[12]:


for col_name in df.columns:
    if(df[col_name].dtype == 'object'):
        df[col_name]= df[col_name].astype('category')
        df[col_name] = df[col_name].cat.codes


# In[13]:


fig, ax = plt.subplots(figsize=(16, 6))
sns.boxplot(x="variable", y="value", data=pd.melt(df),ax=ax)
plt.show()


# In[14]:


standard_deviations = 3
df[df.apply(lambda x: np.abs(x - x.mean()) / x.std() < standard_deviations)
   .all(axis=1)]


# In[15]:


data = df.copy()


# In[16]:


input = df.loc[:, df.columns != 'target']
output= df['target'] 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(input,output,test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=200, random_state=42)
forest.fit(X_train,y_train)


# In[17]:


y_pred_test = forest.predict(X_test)
y_pred_train = forest.predict(X_train)


# In[18]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_test, target_names=val.keys()))
print(classification_report(y_train, y_pred_train, target_names=val.keys()))


# In[19]:


y_prob_test = list(forest.predict_proba(X_test))
y_prob_train = list(forest.predict_proba(X_train))


# In[20]:


y_prob_train.extend(y_prob_test)
prob_class = list(val.keys())


# In[21]:


new_class = []
for prob in y_prob_train:
    str1 = str(round(prob[0],1))+'-'+prob_class[0]+"-"+str(round(prob[1],1))+"-"+prob_class[1]
    new_class.append(str1)


# In[22]:


data['new_class'] = new_class

data.head()


# In[23]:


def evaluate(model):
    print("Train Accuracy :", accuracy_score(y_train, model.predict(x_train)))
    print("Train Confusion Matrix:")
    print(confusion_matrix(y_train, model.predict(x_train)))
    print("Train classification report")
    print(classification_report(y_train, model.predict(x_train)))
    print("-"*50)
    print("Test Accuracy :", accuracy_score(y_test, model.predict(x_test)))
    print("Test Confusion Matrix:")
    print(confusion_matrix(y_test, model.predict(x_test)))
    print("Test classification report")
    print(classification_report(y_test, model.predict(x_test)))


# In[24]:


data.drop('target',axis=1,inplace=True)

for col_name in data.columns:
    if(data[col_name].dtype == 'object'):
        data[col_name]= data[col_name].astype('category')
        data[col_name] = data[col_name].cat.codes

data.head()


# In[25]:


x = data.loc[:, data.columns != 'new_class']
y= data['new_class']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled = scaler.fit_transform(x)

from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test=tts(scaled,y,test_size=0.80,random_state=0)

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(max_depth = 2)
dtree.fit(x_train, y_train)


# In[26]:


from sklearn.metrics import confusion_matrix
test_pred = dtree.predict(x_test)
cm = confusion_matrix(y_test, test_pred)

cm


# In[27]:


print(classification_report(y_test, test_pred))


# In[28]:


params = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'criterion': ["gini", "entropy"]
}

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator=dtree, 
                           param_grid=params, 
                           cv=4, n_jobs=-1, verbose=1, scoring = "accuracy")

grid_search.fit(x_train, y_train)


# In[29]:


grid_search.best_estimator_


# In[30]:


evaluate(grid_search.best_estimator_)


# In[34]:


import tabulate
conclusion = [['Model', 'Precision','Recall', 'F1score', 'Accuracy'],
              ['RandomForest', 0.99, 0.99, 0.99, 0.99],
              ['DecisionTree', 0.05, 0.07 ,0.04, 0.43],
              ['DecisionTree after hyperparamter tunning', 0.03, 0.07 ,0.04, 0.43]
]
print(tabulate.tabulate(conclusion, tablefmt='fancy_grid'))


# In[35]:


Rf=[0.99, 0.99, 0.99, 0.99]
DT= [ 0.05, 0.07 ,0.04, 0.43]
DT_tune=[0.03, 0.07 ,0.04, 0.43]


# In[36]:


import matplotlib.pyplot as plt
import numpy as np


width = 0.1

r1 = np.arange(4)
r4 = [i + width for i in r1]
r5 = [i + width for i in r4]

plt.bar(r1, Rf, color='#375e97', width=width, label='Rf')


plt.bar(r4, DT, color='#3f681c', width=width, label='DT')
plt.bar(r5, DT_tune, color='#f18d9e', width=width, label='DT_tune')

plt.ylim(top=1.1)  # adjust the top leaving bottom unchanged
plt.xlabel('Performance Measures', fontsize=16)
plt.ylabel('% Values', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=16)
plt.title('Test Results')

plt.xticks([r + 2.5*width for r in range(4)], [ 'Prec', 'Recal','f1 score','Acc'])

plt.legend(loc=2 , ncol=2, fontsize=12)
plt.show()

plt.savefig("output.jpg")


# In[ ]:




