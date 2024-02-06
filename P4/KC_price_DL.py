#!/usr/bin/env python
# coding: utf-8

# # Kings County WA House Price Dataset
# # 2조 : 김현솔, 박기열, 오세연, 최민성

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


# <img src='https://raw.githubusercontent.com/Choe-minsung/project-study/main/P3/src/KCC.png' width=400/>

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


plt.figure(figsize = (8,8))

sns.heatmap(df.corr(), annot = True, fmt = '.2f')
plt.show()


# ### PRICE(집값) 과의 상관계수 TOP 5
# - 1위 : sqft_living (0.7)
# - 2위 : grade (0.67)
# - 3위 : sqft_above (0.61)
# - 4위 : sqft_living15 (0.59)
# - 5위 : bathrooms (0.53)

# In[32]:


# 집값과 용도 별 sqft 면적(4개 컬럼)간 scatterplot

fig = plt.figure(figsize=(16, 8))

fig.add_subplot(2, 2, 1)
sns.scatterplot(x=df['sqft_living'], y=df['price'])

fig.add_subplot(2, 2, 2)
sns.scatterplot(x=df['sqft_lot'], y=df['price'])

fig.add_subplot(2, 2, 3)
sns.scatterplot(x=df['sqft_above'], y=df['price'])

fig.add_subplot(2, 2, 4)
sns.scatterplot(x=df['sqft_basement'], y=df['price'])

plt.show()


# - sqft_lot(단순 대지면적)을 제외하고 대체로 sqft가 넓으면 집값이 상승.

# In[33]:


plt.figure(figsize = (6,8))

plt.subplot(2,1,1)
plt.bar(df['grade'], df['price'])

plt.subplot(2,1,2)
plt.bar(df['bathrooms'], df['price'])

plt.show()


# - 예외값이 존재하지만, 대체로 집의 grade와 화장실 수 가 많을수록 집값이 높다.

# In[34]:


df_sorted = df.sort_values('price', ascending = False)

df_sorted = df_sorted.reset_index(drop = True)

df_sorted


# In[35]:


df_sorted[0:1]


# In[36]:


# price 상위 5개 house
df_sorted[['lat', 'long']][:5]


# In[37]:


# price 하위 5개 house
df_sorted[['lat', 'long']][-5:]


# In[38]:


df.shape[0]


# In[39]:


# price 상위 5개 house 위치

locations = []

for i in range(5):
    locations.append([df_sorted.loc[i, 'lat'], df_sorted.loc[i, 'long']])
    

locations


# In[40]:


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


# - price 1위 house
# - 7,700,000 $
#   
# <img src='https://raw.githubusercontent.com/Choe-minsung/project-study/main/P3/src/price_a.png' width=700/>

# - price 1위 house
# - price history ($) : 7,700,000 / 9,850,000 / 9,500,315 
# 
# <img src='https://raw.githubusercontent.com/Choe-minsung/project-study/main/P4/src/1st_price_house.png' width=700/>

# - 실제 Seattle의 집값트렌트 탐색 후, 추가 인사이트 디벨롭

# # 데이터 전처리

# In[43]:


# id, date, yr_built, yr_renovated, zipcode, lat, long, sqft_living15, sqft_lot15 컬럼 삭제
df = df.drop(['id', 'date', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15'], axis = 1)

df.head(2)


# In[44]:


# 'grade' 컬럼 병합 (0 ~ 7)
df.loc[df['grade'] <= 4,'grade'] = 4
df.loc[df['grade'] >= 11,'grade'] = 11

df['grade'] = df['grade'] - 4


# In[45]:


target = 'price'

x = df.drop(target, axis = 1)
y = df[target]


# In[46]:


# train, test셋 split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

# 스케일링
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# # 추가 ML
# ### CatBoost vs XGB, LGMB...
# - Level-wise tree 방식 : 대칭적 트리구조, predict 산출 시간 감소
# - Ordered Boosting 방식 : 일부 data의 잔차를 기반으로 다음 data에 적용시켜 모델 update
# - Random Permutation 방식 : 각 data 잔차 계산시, 매번 shuffle

# <img src='https://raw.githubusercontent.com/Choe-minsung/project-study/main/P4/src/boost_models.png' width=700/>

# In[63]:


from sklearn.metrics import *


# In[47]:


get_ipython().system('pip install catboost')


# In[48]:


# 불러오기
from catboost import CatBoostRegressor

# 선언하기
model = CatBoostRegressor(learning_rate=0.1,
                          depth=4)

# 성능예측
model.fit(x_train, y_train)

# 예측하기
y_pred = model.predict(x_test)

y_pred


# ========================================
# 
# Linear Regression 0.596  
# KNN 0.594  
# Decision Tree 0.533  
# Random Forest 0.589  
# Gradient Boosting Regressor 0.666  
# XGB 0.656  
# LGBM 0.676  
# **CB 0.679**  
# 
# ========================================  
#   
# r2_score: **CB** > LGBM > GB > XGB > LR > KNN > RF > DT

# # DL Modeling

# In[53]:


import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam


# In[176]:


x_train.shape


# ### DL 과적합 방지
# - **Dropout** : 전체 weight를 계산에 참여시키는 것이 아닐 layer에 포함된 weight 중에서 일부만 참여시키는 것
# - **Batch Nomalization** : 배치 정규화는 평균과 분산을 조정하는 과정이 별도의 과정으로 떼어진 것이 아니라, 신경망 안에 포함되어 학습 시 평균과 분산을 조정하는 과정

# In[177]:


clear_session()

il = Input(shape = (11,))
hl = Flatten()(il)

hl = Dense(32, activation = 'relu')(hl)
hl = BatchNormalization()(hl)
hl = Dropout(0.25)(hl)

hl = Dense(16, activation = 'relu')(hl)
hl = BatchNormalization()(hl)
hl = Dropout(0.25)(hl)

hl = Dense(8, activation = 'relu')(hl)
hl = BatchNormalization()(hl)
hl = Dropout(0.25)(hl)

ol = Dense(1, activation = 'linear')(hl)

# model 선언
model = Model(il, ol)

model.summary()


# In[178]:


model.compile( optimizer = 'adam', loss = 'mse', metrics = ['mse'])


# In[179]:


history = model.fit(x_train, y_train, epochs = 10, verbose = 1, validation_data = (x_test, y_test))


# In[180]:


plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.xlabel('Epochs')
plt.ylabel('mse')
plt.legend(['mse', 'val_mse'])
plt.show()


# ### ANN(Artificial Neural Network) vs DNN(Deep Neural Network)
# - ANN의 HL의 갯수 늘리면 -> DNN

# ### ANN

# In[57]:


# ANN 모델 만들기
ann_model = Sequential()
ann_model.add(Dense(64, activation='relu', input_shape=(11,)))
ann_model.add(Dense(32, activation='relu'))
ann_model.add(Dense(1))  # 출력 레이어
 
# 모델 컴파일
ann_model.compile(optimizer='adam', loss='mean_squared_error')
 
# 모델 훈련
history = ann_model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test), verbose=2)
 
# 모델 학습 결과 시각화
plt.figure(figsize=(10, 6))
 
# 훈련 및 검증 손실 플롯
plt.plot(history.history['loss'], label='Training Loss',marker = '.')
plt.plot(history.history['val_loss'], label='Validation Loss',marker = '.')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
 
plt.show()
 


# - epoch 추가

# In[60]:


clear_session()
 
# ANN 모델 만들기
ann_model2 = Sequential()
ann_model2.add(Dense(64, activation='relu', input_shape=(11,)))
ann_model2.add(Dense(32, activation='relu'))
ann_model2.add(Dense(1))  # 출력 레이어
 
# 모델 컴파일
ann_model2.compile(optimizer='adam', loss='mean_squared_error')
 
# 모델 훈련
history = ann_model2.fit(x_train, y_train, epochs=30, batch_size=32, validation_data=(x_test, y_test), verbose=2)
 
# 모델 학습 결과 시각화
plt.figure(figsize=(10, 6))
 
# 훈련 및 검증 손실 플롯
plt.plot(history.history['loss'], label='Training Loss',marker = '.')
plt.plot(history.history['val_loss'], label='Validation Loss',marker = '.')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
 
plt.show()
 


# In[64]:


# epochs 증가 전/후 성능비교

pred =ann_model.predict(x_test)
 
print(mean_squared_error(y_test, pred, squared = False))
print(mean_absolute_error(y_test, pred))
 
pred =ann_model2.predict(x_test)
 
print(mean_squared_error(y_test, pred, squared = False))
print(mean_absolute_error(y_test, pred))


# ### DNN 모델

# In[58]:


# DNN 모델 만들기 (더 깊은 신경망)
dnn_model = Sequential()
dnn_model.add(Dense(128, activation='relu', input_shape=(11,)))
dnn_model.add(Dense(64, activation='relu'))
dnn_model.add(Dense(32, activation='relu'))
dnn_model.add(Dense(1))  # 출력 레이어
 
# 모델 컴파일
dnn_model.compile(optimizer='adam', loss='mean_squared_error')
 
# 모델 훈련
dnn_model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test), verbose=2)
 
# 모델 훈련
history_dnn = dnn_model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test), verbose=2)

# 모델 학습 결과 시각화
plt.figure(figsize=(12, 6))
 
# 훈련 및 검증 손실 플롯
plt.subplot(1, 2, 1)
plt.plot(history_dnn.history['loss'], label='Training Loss',marker = '.')
plt.plot(history_dnn.history['val_loss'], label='Validation Loss',marker = '.')
plt.title('Training and Validation Loss (DNN Model)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
 
plt.show()


# - ANN2 모델로 실제 집값 예측 

# In[72]:


# 실제 집값을 예측

new_data = pd.DataFrame({
    'bedrooms': [3],
    'bathrooms': [2],
    'sqft_living': [2000],
    'sqft_lot': [5000],
    'floors': [2],
    'waterfront' : [1],
    'view': [1],
    'condition': [3],
    'grade': [7],
    'sqft_above': [1500],
    'sqft_basement': [500]
    
})
 
# 새로운 데이터에도 동일한 전처리 적용
new_data_scaled = scaler.transform(new_data)
 
# 성능이 더 좋은 ANN2 모델을 사용하여 집값 예측
predicted_price = ann_model2.predict(new_data_scaled)
 
print("예측 집값:", predicted_price[0][0])


# # Kaggle 1 : House Property Sales (Seattle)
# ### 2022년 시애틀 주택 가격 datasets

# In[181]:


df = pd.read_csv('real_estate_seattle.csv')

df


# In[182]:


df = df[['beds', 'baths', 'price']]

df


# In[183]:


target = 'price'

x = df.drop(target, axis = 1)
y = df[target]
# train, test셋 split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)


# In[184]:


# 스케일링

# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()

# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)


# In[185]:


x_train.shape


# In[186]:


# 딥러닝 모델만들어서 1위 price 집의 seattle 2022년의 평균 가격 적정한지?
clear_session()

il = Input(shape = (2,))
hl = Flatten()(il)

hl = Dense(32, activation = 'relu')(hl)
hl = BatchNormalization()(hl)
hl = Dropout(0.25)(hl)

hl = Dense(16, activation = 'relu')(hl)
hl = BatchNormalization()(hl)
hl = Dropout(0.25)(hl)

hl = Dense(8, activation = 'relu')(hl)
hl = BatchNormalization()(hl)
hl = Dropout(0.25)(hl)

ol = Dense(1, activation = 'linear')(hl)

# model 선언
model = Model(il, ol)

model.summary()


# In[187]:


model.compile( optimizer = 'adam', loss = 'mse', metrics = ['mse'])


# In[188]:


history = model.fit(x_train, y_train, epochs = 100, verbose = 1, validation_data = (x_test, y_test))


# In[189]:


x_test


# In[190]:


y_pred = model.predict([[1, 2]])

y_pred


# - **Seattle**의 침실 3개, 화장실 5개인 집의 예측가격 : **20324 $**

# # Kaggle 2 : House Property Price Prediction (Time Series)
# ### 2007-2019년 집값 시계열 데이터 datasets

# - LSTM : RNN의 **장기 문맥 의존성** 해결

# <img src='https://raw.githubusercontent.com/Choe-minsung/project-study/main/P4/src/lstm.png' width=700/>

# In[191]:


# LSTM import
from keras.layers import LSTM


# In[192]:


df = pd.read_csv('sales_time.csv')

df


# In[193]:


# 거래날짜 컬럼 datetime 형변환
df['datesold'] = pd.to_datetime(df['datesold'])

df.info()


# In[194]:


df['datesold'] = df.loc[df['datesold'].dt.year >= 2014, 'datesold']

df = df.dropna(axis = 0)

df.drop(['postcode', 'propertyType'], axis = 1)


# In[195]:


# 전처리

scaler = MinMaxScaler()

data_scaled = scaler.fit_transform(df['price'].values.reshape(-1, 1))

sequence_length = 10  #10 단위기간 후

X = []  # 현재 집값
y = []  # 미래 집값

for i in range(len(data_scaled) - sequence_length):
    X.append(data_scaled[i:i+sequence_length])
    y.append(df['price'].iloc[i+sequence_length])

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# LSTM 모델
model = Sequential()
model.add(LSTM(50, input_shape=(sequence_length, X.shape[2]), activation = 'relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics = ['mse'])


model.fit(X_train, y_train, epochs = 30, batch_size=32)

pred = model.predict(X_test)



# In[196]:


# 시각화

plt.figure(figsize=(12, 6))

plt.plot(y_test, label='실제 가격')
plt.plot(pred, label='예측 가격')
plt.legend()
plt.show()


# In[197]:


# 미래 집값

future_data = data_scaled[-sequence_length:].reshape(1, -1, X.shape[2])

future_price = model.predict(future_data)

future_price = scaler.inverse_transform(future_price)

future_price


# In[18]:


df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].apply(lambda date:date.month)
df['year'] = df['date'].apply(lambda date:date.year)

fig = plt.figure(figsize=(10, 5))

fig.add_subplot(1, 2, 1)
df.groupby('month').mean()['price'].plot()

fig.add_subplot(1, 2, 2)
df.groupby('year').mean()['price'].plot()


# - Target Region의 집값 움직임을 파악하여 시장 상황에 맞는 지역별로 차별화된 주택정책을 수립 및 진행하여 정책의 효율성을 높인다.
# - 투자자의 입장에서 투자지역의 특징을 고려하려 이상치 값을 최대한 제외하고 보다 과거의 집값 정보를 바탕으로 예측된 가격을 통해 합리적인 투자를 할 수 있을 것으로 기대된다.

# ### Seattle, WA Housing Market 통계

# <img src='https://raw.githubusercontent.com/Choe-minsung/project-study/main/P4/src/seattle_market.png' width=700/>

# - Seattle의 평균 집값은 **800,000 $**

# ## <span style="background-color: #FFB6C1">결론 및 인사이트</span>
# - 집값에 영향을 미치는 가장 중요한 요소는 grade, sqft_living이다.
# - 또한, 집이 거래될 때 제곱미터당 일정 금액의 가격이 책정되기 때문에 'sqft_living'이 높은 상관성을 보인다고 예측할 수 있다.
# - 등급의 경우도 집의 전반적 상태를 표현하는 것일 수도 있지만 집값에 대한 소비자의 심리를 이용한 지표로 생각하였다.
# - Seattle의 물가 상승률, 인프라 접근성, 거주자 만족도 등을 모두 고려하려 계약 또는 투자를 하는 것이 합리적이다.

# ### 추가: lat, long
# - 데이터 전처리 과정에서 lat과 long 컬럼을 삭제한 뒤 모델링을 진행하였지만, 삭제하지 않고 진행하게 되면 lat과 long이 상대적으로 높은 상관성을 보인다.
# - lat(위도)와 long(경도)의 조합은 집의 위치를 나타내기 때문에 집값에 영향을 미치게 된다.
# - 대중교통이 편리한 위치, 학군이 좋은 위치는 일반적으로 더 높은 가격에 판매되고, 공원이나 호수 근처의 장소는 주변 환경에 비해 가격이 높기 때문에 lat과 long이 높게 산출된 것으로 보인다.

# - 출처  
# https://maps.google.com  
# https://sites.google.com/view/vinegarhill-datalabs/introduction-to-machine-learning/kings-county-house-prices  
# https://www.kaggle.com/datasets/htagholdings/property-sales/data   
# https://www.kaggle.com/datasets/samuelcortinhas/house-price-prediction-seattle/  
# https://www.zillow.com/homes/1137-Harvard-Ave-E-Seattle,-WA-98102_rb/49005970_zpid/  
# https://www.redfin.com/city/16163/WA/Seattle/housing-market#top10
