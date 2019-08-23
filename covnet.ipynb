import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

Name = "Cat-vs-Dog-cnn64x2-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(Name))

X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

model = Sequential()
model.add(Conv2D( 64 , (3,3), input_shape = X.shape[1:]) )
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("sigmoid"))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(optimizer="SGD",loss="mean_squared_error",metrics=["accuracy"])

model.fit(X,y,batch_size = 32,validation_split=0.1,epochs = 10,callbacks=[tensorboard])
