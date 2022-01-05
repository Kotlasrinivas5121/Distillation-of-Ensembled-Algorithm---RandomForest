#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score


# In[4]:


df=pd.read_csv('/content/LasVegasTripAdvisorReviews-Dataset.csv',sep=';')


# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


df.dtypes


# In[8]:


df.Score.value_counts().plot(kind="bar")


# In[13]:


df.Score=df.Score.astype(str)
val = df.Score.value_counts()


# In[9]:


import seaborn as sns

corr = df.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)


# In[10]:


df.isnull().sum()


# In[11]:


df.hist(bins=50, figsize=(20,15))
plt.show();


# In[14]:


for col_name in df.columns:
    if(df[col_name].dtype == 'object'):
        df[col_name]= df[col_name].astype('category')
        df[col_name] = df[col_name].cat.codes


# In[15]:


fig, ax = plt.subplots(figsize=(26, 6))
sns.boxplot(x="variable", y="value", data=pd.melt(df),ax=ax)
plt.show()


# In[16]:


standard_deviations = 3
df[df.apply(lambda x: np.abs(x - x.mean()) / x.std() < standard_deviations)
   .all(axis=1)]


# In[17]:


data = df.copy()


# In[25]:


input = df.loc[:, df.columns != 'Score']
output= df['Score'] 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(input,output,test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=200, random_state=42)
forest.fit(X_train,y_train)


# In[26]:


y_pred_test = forest.predict(X_test)
y_pred_train = forest.predict(X_train)


# In[27]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_test, target_names=val.keys()))
print(classification_report(y_train, y_pred_train, target_names=val.keys()))


# In[28]:


y_prob_test = list(forest.predict_proba(X_test))
y_prob_train = list(forest.predict_proba(X_train))


# In[29]:


y_prob_train.extend(y_prob_test)


# In[30]:


prob_class = list(val.keys())


# In[31]:


new_class = []
for prob in y_prob_train:
    str1 = str(round(prob[0],1))+'-'+prob_class[0]+"-"+str(round(prob[1],1))+"-"+prob_class[1]
    new_class.append(str1)


# In[32]:


data['new_class'] = new_class

data.head()


# In[33]:


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


# In[34]:


data.drop('Score',axis=1,inplace=True)

for col_name in data.columns:
    if(data[col_name].dtype == 'object'):
        data[col_name]= data[col_name].astype('category')
        data[col_name] = data[col_name].cat.codes

data.head()


# In[35]:


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


# In[36]:


from sklearn.metrics import confusion_matrix
test_pred = dtree.predict(x_test)
cm = confusion_matrix(y_test, test_pred)

cm


# In[37]:


print(classification_report(y_test, test_pred))


# In[38]:


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


# In[39]:


grid_search.best_estimator_


# In[40]:


evaluate(grid_search.best_estimator_)


# In[41]:


import tabulate
conclusion = [['Model', 'Precision','Recall', 'F1score', 'Accuracy'],
              ['RandomForest', 0.38, 0.24, 0.22, 0.47],
              ['DecisionTree', 0.05, 0.07 ,0.06, 0.77],
              ['DecisionTree after hyperparamter tunning', 0.06, 0.07 ,0.06, 0.77]
]
print(tabulate.tabulate(conclusion, tablefmt='fancy_grid'))


# In[42]:


Rf=[ 0.38, 0.24, 0.22, 0.47]
DT= [0.05, 0.07 ,0.06, 0.77]
DT_tune=[ 0.06, 0.07 ,0.06, 0.77]


# In[43]:


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




