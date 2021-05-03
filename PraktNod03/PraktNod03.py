#!/usr/bin/env python
# coding: utf-8

# <img src="Letters.png" width=324 height=332 />

# In[ ]:


burti = [
    ['A',[0,1,0,1,0,1,1,1,1,1,0,1,1,0,1],[1,0,0]],
    ['E',[1,1,1,1,0,0,1,1,0,1,0,0,1,1,1],[1,0,0]],
    ['D',[1,1,0,1,0,1,1,0,1,1,0,1,1,1,0],[1,0,0]],
    ['B',[1,1,0,1,0,1,1,1,0,1,0,1,1,1,0],[0,1,0]],
    ['C',[1,1,1,1,0,0,1,0,0,1,0,0,1,1,1],[0,1,0]],
    ['F',[1,1,1,1,0,0,1,1,0,1,0,0,1,0,0],[0,1,0]],
    ['G',[1,1,1,1,0,0,1,0,1,1,0,1,1,1,1],[0,0,1]],
    ['H',[1,0,1,1,0,1,1,1,1,1,0,1,1,0,1],[0,0,1]],
    ['I',[0,1,0,0,1,0,0,1,0,0,1,0,0,1,0],[0,0,1]]
]


# In[ ]:


print(burti)


# In[ ]:


import numpy as np

lc = 0.1  #Apmācības koeficients

class Neirons:
    
    def __init__(self):
        self.w = np.zeros(?) #<-- Izlabot, alt?
        self.w0 = 0
    
    def classify(self, x):
        net = np.dot(x, self.w)
        net += self.w0
        y = ? if net < ? else ?  #<-- Izlabot
        return y
    
    def learn(self, x, z):
        y = self.classify(x)
        delta = z-y
        if delta != 0:
            for i in range(0, len(self.w)):
                self.w[i]+= ?  #<-- Izlabot
            self.w0+= ?  #<-- Izlabot


# In[ ]:


n1 = Neirons();
print(n1.classify(burti[0][1]))

for eph in range(0, ?):  #<-- Izlabot
    for bi in range(0, len(burti)):
        n1.learn(burti[bi][1],burti[bi][2][0])

for bi in range(0, len(burti)):
    print(burti[bi][0]+" "+str(n1.classify(burti[bi][1])))
    


# In[ ]:


print(n1.w)
print(n2.w)
print(n3.w)


# In[ ]:


from sklearn.datasets import load_iris

iris = load_iris()

print(iris)


# In[ ]:


import tensorflow as tf
#  from keras.layers import Dense
#  from keras.models import Sequential

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1, activation='sigmoid', input_shape = (len(iris.data[0]), )))

model.compile(tf.optimizers.RMSprop(0.001), loss='mse')

model.fit(iris.data, iris.target, epochs=100)

predictions = model.predict(iris.data)


# In[ ]:


for di in range(0, len(iris.data)):
    print(
        str(iris.data[di][0])+"\t"+
        str(iris.data[di][1])+"\t"+
        str(iris.data[di][2])+"\t"+
        str(iris.data[di][3])+"\t"+
        str(iris.target[di])+"\t"+
        str(predictions[di][0])
    )


# In[ ]:




