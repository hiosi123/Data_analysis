#!/usr/bin/env python
# coding: utf-8

# 서울 따릉이 자전거 분석
# - 1시간 전 날씨 정보를 이용하여 따릉이 자전거의 대여수를 분석
# 
# 분석 목적
# - 자전거 대여수를 예측하여, 자전거 점검을 위해 필요한 자전거 적정량 회수
# - 날씨에 따른 자전거 대여 매출 분석
# 
# 사용한 코딩
# - 파이썬 머신러닝

# In[1]:


import pandas as pd


# In[2]:


Test_data = pd.read_csv('D:/python_study/project/따릉이/test.csv', encoding = 'cp949')


# 테스트할 날씨 데이터:
# - 날씨 데이터 종류: 온도, 강수량, 풍속, 습도, 가시거리, 오존, pm2.5 (초미세 먼지), pm10 (미세 먼지)

# In[3]:


Test_data.head()


# 데이터 수
#  - 715개의 날씨 데이터

# In[4]:


Test_data.shape


# 데이터 정제 작업 

# In[5]:


Test_data.isna().sum()


# In[6]:


Test_data.fillna(Test_data['hour_bef_temperature'].mean(), inplace = True)
Test_data.fillna(Test_data['hour_bef_precipitation'].mean(), inplace = True)
Test_data.fillna(Test_data['hour_bef_windspeed'].mean(), inplace = True)
Test_data.fillna(Test_data['hour_bef_humidity'].mean(), inplace = True)
Test_data.fillna(Test_data['hour_bef_visibility'].mean(), inplace = True)
Test_data.fillna(Test_data['hour_bef_ozone'].mean(), inplace = True)
Test_data.fillna(Test_data['hour_bef_pm10'].mean(), inplace = True)
Test_data.fillna(Test_data['hour_bef_pm2.5'].mean(), inplace = True)


# In[7]:


Test_data.isna().sum()


# In[8]:


Test_data.info()


# 학습시킬 날씨 데이터
# - 날씨 데이터와, 대여수 포함

# In[9]:


Train_data = pd.read_csv('D:\python_study\project\따릉이/train.csv',encoding = 'cp949')


# In[10]:


Train_data.head()


# In[11]:


Train_data.shape


# 학습 날씨 데이터 정제

# In[12]:


Train_data.isna().sum()


# In[13]:


Train_data.fillna(Train_data['hour_bef_temperature'].mean(), inplace = True)
Train_data.fillna(Train_data['hour_bef_precipitation'].mean(), inplace = True)
Train_data.fillna(Train_data['hour_bef_windspeed'].mean(), inplace = True)
Train_data.fillna(Train_data['hour_bef_humidity'].mean(), inplace = True)
Train_data.fillna(Train_data['hour_bef_visibility'].mean(), inplace = True)
Train_data.fillna(Train_data['hour_bef_ozone'].mean(), inplace = True)
Train_data.fillna(Train_data['hour_bef_pm10'].mean(), inplace = True)
Train_data.fillna(Train_data['hour_bef_pm2.5'].mean(), inplace = True)


# In[14]:


Train_data.isna().sum()


# In[15]:


X= Train_data.drop(['id','count'],axis = 1)
X


# In[16]:


y = Train_data['count']
y


# 학습 코딩 시작

# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


from sklearn.neighbors import KNeighborsRegressor as KNN


# In[19]:


from sklearn.metrics import accuracy_score


# In[20]:


import numpy as np


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0 )


# In[22]:


reg = KNN(n_neighbors = 4)


# In[23]:


reg.fit(X_train, y_train)


# In[24]:


print('test set r^2: {:.2f}'.format(reg.score(X_test,y_test)))


# In[25]:


print('train set r^2: {:.2f}'.format(reg.score(X_train, y_train)))


# In[26]:


from sklearn.ensemble import GradientBoostingRegressor


# In[27]:


from sklearn.metrics import mean_squared_error as MSE 


# In[28]:


gbt = GradientBoostingRegressor(n_estimators = 300,subsample = 0.9, max_features = 0.75, max_depth = 4, random_state = 0)


# In[29]:


gbt.fit(X_train, y_train)


# In[30]:


y_pred = gbt.predict(X_test)


# In[31]:


rmse_test = MSE(y_test, y_pred) **(1/2)


# 학습 결과 
# - 오차범위 36.77

# In[32]:


print('RMSE_test: {:.2f}'.format(rmse_test))


# 예측 값, 실제 값 비교!

# In[51]:


list(zip(y_pred, y_test))


# 테스트 데이터를 이용하여 예측값 만들어 내기

# In[34]:


Test_data.head()


# In[35]:


Test_data


# In[36]:


Test_data.drop('id',axis = 1, inplace = True)


# In[37]:


y_pred_Test = gbt.predict(Test_data)


# In[38]:


print(y_pred_Test)


# In[39]:


Test_data['count'] = y_pred_Test


# 결과 도출, 자전거 대여수 예측 값 입력

# In[40]:


Test_data.head()


# In[41]:


submission = pd.read_csv('D:/python_study/project/따릉이/submission.csv',encoding = 'cp949')


# In[42]:


submission.head()


# In[43]:


submission['count'] = y_pred_Test


# In[44]:


submission.head()


# In[45]:


submission.reset_index(drop = True, inplace = True)


# In[46]:


submission.head()


# In[47]:


submission['count'] = submission['count'].astype(int)


# In[48]:


submission.to_csv('D:/python_study/project/따릉이/submission.csv',encoding = 'cp949')

