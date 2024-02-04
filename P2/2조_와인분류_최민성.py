#!/usr/bin/env python
# coding: utf-8

# # Wine classification
# ## 2조 - 김진서 박금령 최민성

# In[1]:


# 라이브러리 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from PIL import Image

warnings.filterwarnings(action='ignore')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")

plt.rc('font', family='Malgun Gothic') # 한글폰트 사용


# <img src='https://raw.githubusercontent.com/Choe-minsung/project-study/main/P2/src/wine.png' width=800/>

# In[3]:


data = pd.read_csv('wine_dataset.csv')

data.head()


# ## 데이터탐색

# |	변수명	|	설명	|	구분	|
# |----|----|----|
# |		**style**	 	|	 *red or white*	|	**Target**	|
# |		**quality**	 	|	 *와인의 질*	|	**Target**	|
# |		fixed_acidity 	|	 산도	|	feature	|
# |	volatile_acidity	 	|	 휘발성 산	|	feature	|
# |	citric_acid 	|	 시트르산	|	feature	|
# |	residual_sugar	 	|	 잔여 당분	|	feature	|
# |		chlorides 	|	 염화물	|	feature	 |
# |	free_sulfur_dioxide	 	|	 독립 이산화황	|	feature	|
# |	total_sulfur_dioxide	 	|	 총 이산화황	|	feature	|
# |		density 	|	 밀도	|	feature	|
# |		pH	 	|	 수소이온 농도	|	feature	|
# |		sulphates 	|	 황산염	|	feature	|
# |		alcohol	 	|	 도수	|	feature	|
# 
# 
# 
# 

# In[4]:


data.info()


# In[5]:


data.isna().sum()


# In[6]:


data['style'].value_counts()


# In[7]:


data['quality'].value_counts()


# ## 데이터 전처리

# ### quality classification 오류해결
# - train하는 모든 features의 type이 numeric
# - style컬럼을 number형 변환

# In[8]:


# style컬럼 object-> number 형 변환
data.loc[data['style'] == 'red', 'style'] = 0
data.loc[data['style'] == 'white', 'style'] = 1


# In[9]:


data['style'] = data['style'].astype('int64')


# ### XGB model 오류해결
# - Invalid classes inferred from unique values of `y`.  Expected: [0 1 2 3 4 5 6], got [3 4 5 6 7 8 9]
# - quality의 범주값 0부터 시작하도록 변경

# In[10]:


data['quality'] = data['quality'] - 3

data['quality'].value_counts()


# In[11]:


data.info()


# In[12]:


sns.histplot(x = data['quality'], hue = data['style'])


# In[13]:


plt.figure(figsize = (10, 10))
sns.heatmap(data.corr(), annot = True, fmt = '.2f')
plt.show()


# ## Classification

# ## 1. style classification
# - target : style(red or white)
# - 즉, 0 or 1로 표현할 수 있으므로 LogisticRegression 적합

# In[14]:


# x/y split
target = 'style'

x = data.drop(target, axis = 1)
y = data[target]


# In[15]:


# train/test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)


# In[16]:


# scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[17]:


data.head(2)


# In[18]:


# model : LogisticRegression
from sklearn.linear_model import LogisticRegression

model_s = LogisticRegression()

model_s.fit(x_train, y_train)

y_pred = model_s.predict(x_test)


# In[19]:


# verification
from sklearn.metrics import *

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[21]:


model_s.predict_proba(x_test)


# ## 2. quality classification
# - target : quality (3 ~ 9)
# - style 컬럼을 number화(전처리 o)
# - 분류모델들 중, 최적 model 찾기

# In[22]:


# x/y split
target = 'quality'

x = data.drop(target, axis = 1)
y = data[target]


# In[23]:


# train/test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)


# In[24]:


# scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[25]:


# KNN hyperparameter tuning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# max_depth : 1 ~ 10
param = {'n_neighbors' : range(1, 21)}

# model

model = KNeighborsClassifier()

model = GridSearchCV(model, param, cv = 5)

model.fit(x_train, y_train)

print(f'최적 n_neighbors : {model.best_params_}')


# In[26]:


# best_params_ KNN
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors = 1)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)


# In[27]:


# verification
from sklearn.metrics import *

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

result = {}
result['KNN'] = accuracy_score(y_test, y_pred) 


# In[28]:


# DecisionTree hyperparameter tuning
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# max_depth : 1 ~ 10
param = {'max_depth' : range(1, 11)}

# model

model = DecisionTreeClassifier()

model = GridSearchCV(model, param, cv = 5)

model.fit(x_train, y_train)

print(f'최적 max_depth : {model.best_params_}')


# In[29]:


# best_params_ DecisionTree

model = DecisionTreeClassifier(max_depth = model.best_params_['max_depth'])

model.fit(x_train, y_train)

y_pred = model.predict(x_test)


# In[30]:


# verification
from sklearn.metrics import *

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

result['DecisionTree'] = accuracy_score(y_test, y_pred) 


# In[31]:


# XGB hyperparameter tuning
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# max_depth : 1 ~ 10
param = {'max_depth' : range(1, 11)}

# model

model = XGBClassifier()

model = GridSearchCV(model, param, cv = 5)

model.fit(x_train, y_train)

print(f'최적 max_depth : {model.best_params_}')


# In[32]:


# best_params_ XGB

model_q = XGBClassifier(max_depth = model.best_params_['max_depth'])

model_q.fit(x_train, y_train)

y_pred = model_q.predict(x_test)


# In[33]:


# verification
from sklearn.metrics import *

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

result['XGB'] = accuracy_score(y_test, y_pred) 


# In[34]:


# result 비교
result


# ### model selection Result
# - style classification : LogisticRegression(98% 정확도) 사용
# - quality classificaion : XGBClassifier(66.7% 정확도) 사용

# ## Re-test Perfomance Verification

# ### 1. style verification
# - model 정확도 98%.
# - 100 ~ 6000 행 중, 1000의 step 으로 무작위 추출하여 추정값 비교

# In[54]:


data.head(3)


# In[63]:


data_1 = data.drop('style', axis = 1)


# In[67]:


wines = []

for i in range(100, 6000, 1000):
    wines.append(data_1.loc[i,:].values)
    
wines


# In[68]:


wines = scaler.transform(wines)


# In[69]:


model_s.predict(wines)


# In[75]:


for i in range(100, 6000, 1000):
    print(data['style'][i], end = ' ')


# ### 2. quality verification
# - model 정확도 66.7%.
# - 100 ~ 6000 행 중, 500의 step 으로 무작위 추출하여 추정값 비교

# In[76]:


data_2 = data.drop('quality', axis = 1)


# In[81]:


wines = []

for i in range(100, 6000, 500):
    wines.append(data_2.loc[i,:].values)
    
wines


# In[82]:


wines = scaler.transform(wines)


# In[88]:


pred = model_q.predict(wines)


# In[92]:


val = []
for i in range(100, 6000, 500):
    val.append(data['quality'][i])
    
val == pred


# - **출처**  
# 
# 
# Kaggle - Red & White wine Dataset  
# https://www.kaggle.com/datasets/numberswithkartik/red-white-wine-dataset
