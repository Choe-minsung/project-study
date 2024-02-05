#!/usr/bin/env python
# coding: utf-8

# # Kings County WA House Price Dataset
# # 2조 : 오세연, 황승도, 최민성

# In[25]:


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


# <img src='https://raw.githubusercontent.com/Choe-minsung/project-study/main/P3/src/KCC.png' width=300/>

# In[27]:


df = pd.read_csv('kc_houseprice.csv')

df.head(2)


# # 데이터탐색

# id : A notation for a house
# 
# date: Date house was sold
# 
# price: Price is prediction target
# 
# bedrooms: Number of bedrooms
# 
# bathrooms: Number of bathrooms
# 
# sqft_living: Square footage of the home (생활면적 평방미터)
# 
# sqft_lot: Square footage of the lot (대지면적 평방미터)
# 
# floors :Total floors (levels) in house
# 
# waterfront :House which has a view to a waterfront
# 
# view: Has been viewed
# 
# condition :How good the condition is overall
# 
# grade: overall grade given to the housing unit, based on King County grading system
# 
# sqft_above : Square footage of house apart from basement (지상 생활 면적)
# 
# sqft_basement: Square footage of the basement (지하 생활 면적)
# 
# yr_built : Built Year
# 
# yr_renovated : Year when house was renovated
# 
# zipcode: Zip code (우편번호)
# 
# lat: Latitude coordinate
# 
# long: Longitude coordinate
# 
# sqft_living15 : Living room area in 2015(implies-- some renovations) This might or might not have affected the lotsize area
# 
# sqft_lot15 : LotSize area in 2015(implies-- some renovations)

# In[28]:


df.info()


# In[29]:


df.isna().sum()


# In[30]:


df['price'].value_counts()


# In[31]:


plt.figure(figsize = (10,10))

sns.heatmap(df.corr(), annot = True, fmt = '.2f')
plt.show()


# ### 집값과의 상관계수
# - 1위 : 주거면적 (0.7)
# - 2위 : grade (0.67)

# In[32]:


plt.scatter(df['sqft_living'], df['price'])


# In[33]:


plt.bar(df['grade'], df['price'])


# In[34]:


df_sorted = df.sort_values('price', ascending = False)

df_sorted = df_sorted.reset_index(drop = True)

df_sorted


# In[35]:


# price 상위 5개 house
df_sorted[['lat', 'long']][:5]


# In[36]:


# price 하위 5개 house
df_sorted[['lat', 'long']][-5:]


# In[37]:


df.shape[0]


# In[38]:


# price 상위 5개 house 위치

locations = []

for i in range(5):
    locations.append([df_sorted.loc[i, 'lat'], df_sorted.loc[i, 'long']])
    

locations


# In[39]:


# 상위 price map 마킹
# !pip install folium
import folium

# 지도 중심좌표 (Madrona Park, Seattle)
map_center = [47.607171, -122.283942]

# 지도 객체 생성
m = folium.Map(location=map_center, zoom_start = 11)

# 마커추가
for location in locations:
    folium.Marker(
        location=location,
        icon=folium.Icon(icon="cloud"),  
    ).add_to(m)
    
m


# In[40]:


# price 하위 5개 house 위치

locations = []

for i in range(5):
    locations.append([df_sorted.loc[df.shape[0]-i-1, 'lat'], df_sorted.loc[df.shape[0]-i-1, 'long']])
    

locations


# In[41]:


# 하위 price map 마킹

# 지도 중심좌표 (Madrona Park, Seattle)
map_center = [47.607171, -122.283942]

# 지도 객체 생성
m = folium.Map(location=map_center, zoom_start = 10)

# 마커추가
for location in locations:
    folium.Marker(
        location=location,
        icon=folium.Icon(icon="cloud"),  
    ).add_to(m)
    
m


# - price 1위 house
# - 7,700,000 $
#   
# <img src='https://raw.githubusercontent.com/Choe-minsung/project-study/main/P3/src/price_a.png' width=700/>

# - price -1위 house
# - 75,000 $
#   
# <img src='https://raw.githubusercontent.com/Choe-minsung/project-study/main/P3/src/price_b.png' width=700/>

# # 데이터 전처리

# In[44]:


# id, date, yr_built, yr_renovated, zipcode, lat, long, sqft_living15, sqft_lot15 컬럼 삭제
df = df.drop(['id', 'date', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15'], axis = 1)

df.head(2)


# In[45]:


# 'grade' 컬럼 병합 (0 ~ 7)
df.loc[df['grade'] <= 4,'grade'] = 4
df.loc[df['grade'] >= 11,'grade'] = 11

df['grade'] = df['grade'] - 4


# In[46]:


target = 'price'

x = df.drop(target, axis = 1)
y = df[target]


# In[51]:


# train, test셋 split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

# 스케일링
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# # ML 1. LinearRegression

# In[53]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)


# In[54]:


# 정확도 저장
from sklearn.metrics import *

result = {}

result['LR'] = r2_score(y_test, y_pred)


# In[55]:


result


# In[56]:


a = model.coef_
b = model.intercept_

a, b


# In[57]:


# sqft_living, grade 각각의 가중치
a[2], a[-3]


# ## 회귀선 시각화
# 1. 'sqft_living' 과 target의 회귀선시각화
# 2. 'grade' 과 target의 회귀선시각화

# In[58]:


X = [min(df['sqft_living']), max(df['sqft_living'])]

X


# In[66]:


Y1 = X[0] * a[2] + b
Y2 = X[1] * a[2] + b

Y = [Y1, Y2]

Y


# In[67]:


plt.scatter(df['sqft_living'], df['price']) # 실제값들 scatterplot

plt.plot(X, Y) # 회귀식 plot


# In[61]:


X = [min(df['grade']), max(df['grade'])]

X


# In[62]:


Y1 = X[0] * a[-3] + b
Y2 = X[1] * a[-3] + b

Y = [Y1, Y2]

Y


# In[63]:


plt.scatter(df['grade'], df['price']) # 실제값들 scatterplot

plt.plot(X, Y) # 회귀식 plot


# - sqft_living은 1m**2 증가 떄 마다 **평균집값 140(usd) 상승**
# - grade는 1등급 올라갈 때 마다 **평균집값 92985(usd) 상승**

# # ML2 : ENSENBLE
# 1. Random Forest

# In[77]:


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(max_depth = 5, n_estimators = 100)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)


# In[78]:


result['RF'] = r2_score(y_test, y_pred)

result


# In[79]:


plt.barh(x.columns, model.feature_importances_)
plt.show()


# 2. Gradient Boosting Regressor

# In[123]:


from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(max_depth=5, n_estimators=100, learning_rate=0.1)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)


# In[87]:


plt.barh(x.columns, model.feature_importances_)
plt.show()


# In[124]:


result['GB'] = r2_score(y_test, y_pred)

result


# - r2_score : GB > RF > LR

# ### GradientBoostingRegressor 튜닝

# In[100]:


from sklearn.model_selection import GridSearchCV

# 'max_depth' 파라미터 지정
param = {'max_depth' : range(1, 11)}

model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)

model = GridSearchCV(model, param, cv = 5)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)


# In[101]:


# 최적 파라미터
model.best_params_


# In[127]:


model = GradientBoostingRegressor(max_depth=3, n_estimators=100, learning_rate=0.1)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)


# In[128]:


r2_score(y_test, y_pred)


# ### 집값 예측 Test

# In[154]:


# bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, grade, sqft_above, sqft_basement
my_test = [[2, 3, 1180, 5650, 1, 0, 1, 3, 6, 1180, 11000]]

my_test = scaler.transform(my_test)

print(f'해당 집의 예상가격은 {int(model.predict(my_test)[0])} 달러 입니다.')


# - 출처  
# https://maps.google.com  
# https://sites.google.com/view/vinegarhill-datalabs/introduction-to-machine-learning/kings-county-house-prices
