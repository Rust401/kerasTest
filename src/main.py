from __future__ import print_function
import numpy as np
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from keras.utils import plot_model
import tensorflow as tf
import keras as keras
 

batch_size = 32 #mini_batch_size
nb_epoch = 5 #大循环次数
nb_classes=5
input_shape=6


X_train=np.loadtxt("G:\Desktop\DLModel\data\\train\\test.txt",dtype='float',delimiter='\t',usecols=(0,1,2,3,4,5))
Y_train=np.loadtxt("G:\Desktop\DLModel\data\\train\\test.txt",dtype='float',delimiter='\t',usecols=(6,7,8,9,10))

X_test=np.loadtxt("G:\Desktop\DLModel\data\\train\\test.txt",dtype='float',delimiter='\t',usecols=(0,1,2,3,4,5))
Y_test=np.loadtxt("G:\Desktop\DLModel\data\\train\\test.txt",dtype='float',delimiter='\t',usecols=(6,7,8,9,10))

print(X_train)

""" X_train=np.random.random((1000,1000))
Y_train=np.random.randint(20,size=(1000,20))

X_test=np.random.random((1000,1000))
Y_test=np.random.randint(20,size=(1000,20)) """

model = Sequential()#第一层<br>#Dense就是全连接层
model.add(Dense(128, input_shape=(input_shape,))) #输入维度
model.add(Activation('relu')) #激活函数
model.add(Dense(128)) 
model.add(Activation('relu')) #激活函数
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
#损失函数设置、优化函数，衡量标准
model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer='sgd',
              metrics=[tf.keras.metrics.RootMeanSquaredError()])

history = model.fit(X_train, Y_train,
                    nb_epoch=nb_epoch, batch_size=batch_size,
                    verbose=1, validation_split=0.1)


score = model.evaluate(X_test, Y_test,
                       batch_size=batch_size, verbose=1)



print('Test score:', score[0])
print('Test accuracy:', score[1])

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plot_model(model, to_file='model.png')