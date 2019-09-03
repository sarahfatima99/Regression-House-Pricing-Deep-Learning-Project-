#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier


# In[3]:


dataframe=pandas.read_csv("housing.csv",delim_whitespace=True, header=None )
dataset=dataframe.values
x=dataset[:,0:13]
y=dataset[:,13]


# In[4]:


def build_model():
    model=Sequential()
    model.add(Dense(13,activation='relu',input_shape=(13,)))
    model.add(Dense(1))

    model.compile(optimizer='Adam',loss='mse',metrics=['mse'])
    return model
seed=7
np.random.seed(seed)
estimator=KerasRegressor(build_fn=build_model,epochs=100,batch_size=16,verbose=0)
kfold=KFold(n_splits=10,random_state=seed)
result=cross_val_score(estimator,x,y,cv=kfold)
print("Result:%.2f(%.2f)MSE:"%(result.mean(),result.std()))


# In[14]:


np.random.seed(seed)
estimators=[]
estimators.append(('standardize',StandardScaler()))
estimators.append(('mlp',KerasRegressor(build_fn=build_model,epochs=50,batch_size=50,verbose=0)))
pipeline=Pipeline(estimators)
kfold=KFold(n_splits=10,random_state=seed)
print("Standardize:%.2f(%.2f)MSE"%(result.mean(),result.std()))


# In[6]:


#normalizing the data
eny=y
mean = eny.mean(axis=0)
eny -= mean
std = eny.std(axis=0)
eny /= std
#print(eny)
def build_model():
    model=Sequential()
    model.add(Dense(13,activation='relu',input_shape=(13,)))
    model.add(Dense(1))
    model.compile(optimizer='Adam',loss='mse',metrics=['mse'])
    return model
seed=7
np.random.seed(seed)
estimator=KerasRegressor(build_fn=build_model,epochs=100,batch_size=5,verbose=0)
kfold=KFold(n_splits=10,random_state=seed)
result=cross_val_score(estimator,x,eny,cv=kfold)
print("Result:%.2f(%.2f)MSE:"%(result.mean(),result.std()))


# In[ ]:





# # SMALLER MODEL

# In[ ]:


def build_model():
    model=Sequential()
    model.add(Dense(13,activation='relu',input_shape=(13,)))
    model.add(Dense (6,activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='Adam',loss='mse',metrics=['mae'])
    return model
seed=7
np.random.seed(seed)
estimator=KerasRegressor(build_fn=build_model,epochs=100,batch_size=5,verbose=0)
kfold=KFold(n_splits=10,random_state=seed)
result=cross_val_score(estimator,x,eny,cv=kfold)
print("Result:%.2f(%.2f)MSE:"%(result.mean(),result.std()))


# In[ ]:


def build_model():
    model=Sequential()
    model.add(Dense(13,activation='relu',input_shape=(13,)))
    model.add(Dense (20,activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='Adam',loss='mse',metrics=['mae'])
    return model
np.random.seed(seed)
estimators=[]
estimators.append(('standardize',StandardScaler()))
estimators.append(('mlp',KerasRegressor(build_fn=build_model,epochs=50,batch_size=50,verbose=0)))
pipeline=Pipeline(estimators)
kfold=KFold(n_splits=10,random_state=seed)
print("Wider:%.2f(%.2f)MSE"%(result.mean(),result.std()))


# # LARGER MODEL 

# In[ ]:


def build_model():
    model=Sequential()
    model.add(Dense(13,activation='relu',input_shape=(13,)))
    model.add(Dense (80,activation='relu'))
    model.add(Dense (250,activation='relu'))
    model.add(Dense (420,activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='Adam',loss='mse',metrics=['mae'])
    return model
np.random.seed(seed)
estimators=[]
estimators.append(('standardize',StandardScaler()))
estimators.append(('mlp',KerasRegressor(build_fn=build_model,epochs=500,batch_size=50,verbose=0)))
pipeline=Pipeline(estimators)
kfold=KFold(n_splits=10,random_state=seed)
print("Wider:%.2f(%.2f)MSE"%(result.mean(),result.std()))


# #  TUNNING THE MODEL

# In[ ]:


def build_model():
    model=Sequential()
    model.add(Dense(13,activation='relu',input_shape=(13,)))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    return model
seed=7
np.random.seed(seed)
estimator=KerasRegressor(build_fn=build_model,epochs=100,batch_size=5,verbose=0)
kfold=KFold(n_splits=10,random_state=seed)
result=cross_val_score(estimator,x,y,cv=kfold)
print("Result:%.2f(%.2f)MSE:"%(result.mean(),result.std()))


# # FUNCTIONAL API

# In[16]:


from keras.models import Model
from keras.layers import Input, Dense
def build_model():
    input_tensor = Input(shape=(13,))
    x = Dense(13, activation='relu')(input_tensor)
    output_tensor = Dense(1)(x)
    model = Model(input_tensor, output_tensor)
    model.compile(optimizer='Adam', loss='mse', metrics=['mae'])
    return model
seed=7
np.random.seed(seed)
estimator=KerasRegressor(build_fn=build_model,epochs=100,batch_size=5,verbose=0)
kfold=KFold(n_splits=10,random_state=seed)
result=cross_val_score(estimator,x,y,cv=kfold)
print("Result:%.2f(%.2f)MSE:"%(result.mean(),result.std()))


# #  WITH KERAS

# In[4]:


def build_model():
    model=Sequential()
    model.add(Dense(13,activation='relu',input_shape=(13,)))
    model.add(Dense(1))

    model.compile(optimizer='Adam',loss='mse',metrics=['mse'])
    return model


# In[5]:


np.random.shuffle(dataset)
#normalizing the data
eny=y
mean = eny.mean(axis=0)
eny -= mean
std = eny.std(axis=0)
eny /= std

x.shape
print(y.shape)
data=x[:350]
ldata=eny[:350]


# In[6]:


num_validation_samples = 100

validation_data = data[:num_validation_samples]
validation_label=ldata[:num_validation_samples]
data = data[num_validation_samples:]
ldata=ldata[num_validation_samples:]
training_data = data[:]
training_label=ldata[:]
model = build_model()
model.fit(training_data, training_label, epochs=100, batch_size=5)


# In[7]:


def build_model():
    model=Sequential()
    model.add(Dense(13,activation='relu',input_shape=(13,)))
    model.add(Dense(1))

    model.compile(optimizer='Adam',loss='mse',metrics=['mse'])
    return model
model.fit(validation_data, validation_label, epochs=100, batch_size=5)


# In[34]:


def build_model():
    model=Sequential()
    model.add(Dense(13,activation='relu',input_shape=(13,)))
    model.add(Dense(1))

    model.compile(optimizer='Adam',loss='mse',metrics=['mse'])
    return model
model.fit(x[350:], eny[350:], epochs=100, batch_size=5)


# # MODEL SUBCLASSING

# In[20]:


import keras
class build_model(keras.Model):
    def __init__(self):
        super(build_model,self).__init__()
        inputs = (13,)
        self.dense1=Dense(13,activation='relu')
        self.dense2=Dense(1)
    def call(self,inputs):
        x=self.dense1(inputs)
        return self.dense2(x)
    
def finalModel():
    model=build_model()
    model.compile(optimizer='Adam',loss='mse',metrics=['mae'])
    return model
np.random.seed(seed)
estimators=[]
estimators.append(('standardize',StandardScaler()))
estimators.append(('mlp',KerasRegressor(build_fn=build_model,epochs=50,batch_size=50,verbose=0)))
pipeline=Pipeline(estimators)
kfold=KFold(n_splits=10,random_state=seed)
print("Standardize:%.2f(%.2f)MSE"%(result.mean(),result.std()))


# In[ ]:





# In[ ]:




