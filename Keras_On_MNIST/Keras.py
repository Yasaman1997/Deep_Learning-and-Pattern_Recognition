
# coding: utf-8

# In[1]:


import numpy as np
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[2]:


np.max(X_train)


# In[3]:


X_train = X_train / 255.
X_test = X_test / 255.


# # Network Model

# In[4]:


from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(128, activation='relu', input_shape=(28*28,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))


# # Loss function

# In[5]:


from keras import optimizers

adam = optimizers.Adam(lr=0.01)

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])


# # Optimization

# In[8]:


from keras.utils import np_utils

X = X_train.reshape((-1, 28*28))
y = np_utils.to_categorical(y_train, 10)

X_val = X_test.reshape((-1, 28*28))
y_val = np_utils.to_categorical(y_test, 10)

model.fit(X, y, batch_size=32, epochs=10, verbose=1, validation_data=(X_val, y_val))


# In[9]:


loss, score = model.evaluate(X_val, y_val, verbose=0)
score


# # a Better Approach

# In[10]:


from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[11]:


from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution2D

model = Sequential()

model.add(Convolution2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Convolution2D(16, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))


# In[12]:


from keras import optimizers

adam = optimizers.Adam(lr=0.001)

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])


# In[ ]:


from keras.utils import np_utils

X = X_train.reshape((-1, 28, 28, 1))
y = np_utils.to_categorical(y_train, 10)

X_val = X_test.reshape((-1, 28, 28, 1))
y_val = np_utils.to_categorical(y_test, 10)

model.fit(X, y, batch_size=32, nb_epoch=10, verbose=1, validation_data=(X_val, y_val))


# # Evaluation

# In[ ]:


X = X_test.reshape((-1, 28, 28, 1))
y = np_utils.to_categorical(y_test, 10)

loss, score = model.evaluate(X, y, verbose=0)
score

