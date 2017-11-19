
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np


# In[3]:

df=pd.read_csv('(A)TRAIN_SMS.csv')


# In[5]:

dict={}
count=0
for i in df['Label'].unique():
    count+=1
    dict[i]=count
print(dict)


# In[6]:

df['LabelCode']=df['Label'].map(dict)


# In[9]:

import numpy
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.preprocessing import sequence,text
from keras.layers.embeddings import Embedding
import pandas as pd

X = df['Message']
y = df['LabelCode']

tk = text.Tokenizer(nb_words=200, lower=True)
tk.fit_on_texts(X)

x = tk.texts_to_sequences(X)
numpy.random.seed(7)
print(X.shape)
print(y.shape)

###################################

print (len(tk.word_counts))

###################################
max_len = 80
print ("max_len ", max_len)
print('Pad sequences (samples x time)')

x = sequence.pad_sequences(x, maxlen=max_len)



max_features = 200
model = Sequential()
print('Build model...')

model = Sequential()
model.add(Embedding(max_features, 128, input_length=max_len, dropout=0.2))
model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))
model.add(Dense(3))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["accuracy",])

print(y[0])
# Prepare Y
y2 = []
for i in range(y.shape[0]):
    row = [0 for _ in range(3)]
    row[int(np.squeeze(y[i]))-1] = 1
    y2.append(row)

y = np.array(y2)

model.fit(x, y=y, batch_size=500, nb_epoch=14, verbose=1, validation_split=0.2, shuffle=True)


# In[ ]:



