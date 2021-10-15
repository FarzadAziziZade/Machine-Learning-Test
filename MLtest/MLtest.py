# Farzad Azizi Zade
#-------------------------------------------------------
modee=2 # set either you want two or one output
#-------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import StandardScaler
#-------------------------------------------------------
csv_file_name = "datain"
#-------------------------------------------------------
header_list = ["f1", "f2", "f3", "f4", "o5", "o6", "o7"]
data = pd.read_csv('{file_name}.csv'.format(file_name = csv_file_name), names=header_list).replace('?', np.NaN).dropna()
X = data.get(["f1", "f2", "f3", "f4"])
if modee==2:
    y = data.get(["o5", "o6"])
elif modee==1:
    y = data.get(["o7"])
#-------------------------------------------------------
#data.head() # this function shows only the first 5 values
#-------------------------------------------------------
# initialising the Scaler
scaler = MinMaxScaler()
#scaler = minmax_scale()
#scaler = StandardScaler()
#scaler=Normalizer() # it is the best one, but it does not have inverse
# learning the statistical parameters for each of the data and transforming
scaler1=scaler
scaler2=scaler
X = scaler1.fit_transform(X)
y = scaler2.fit_transform(y)
#-------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle = True)
#print(X_train)
#print(X_test)
#print(X_val)
#print(y_train)
#print(y_test)
#print(y_val)
#-------------------------------------------------------
import tensorflow as tf
#-------------------------------------------------------
if modee==2:
    nnn=2
elif modee==1:
    nnn=1 
#-------------------------------------------------------
model = tf.keras.models.Sequential([                 
    tf.keras.layers.Dense(4, activation='tanh', input_dim = 4, name="Dense_1"),
    tf.keras.layers.Dense(8, activation='tanh', name="Dense_2"),
    tf.keras.layers.Dense(2, activation='tanh', name="Output")
], name="Model_1")
#https://keras.io/api/metrics/
model.compile(  optimizer='RMSprop', 
                loss='mean_squared_error', 
                metrics=['accuracy'])
model.summary() #it will print test data and predicted results
#-------------------------------------------------------
history = model.fit(  X_train, 
            y_train, 
            epochs=300,
            validation_data=(X_val, y_val),
            verbose=0)
#By setting verbose 0, 1 or 2 you just say how do you want to 'see' the training progress for each epoch.
#verbose=0 will show you nothing (silent)
#verbose=1 will show you an animated progress bar like this:
# progres_bar [==========================================]
#verbose=2 will just mention the number of epoch like this:
# Epoch 1/300
#-------------------------------------------------------
print(model.evaluate(X_test, y_test, verbose=2))
#-------------------------------------------------------
ee=scaler2.inverse_transform(model.predict(X_test)-y_test)
#-------------------------------------------------------
model.save("NNTest.h5")
#-------------------------------------------------------
import matplotlib.pyplot as plt
acc = history.history['accuracy'] ### YOUR CODE HERE
val_acc = history.history['val_accuracy'] ### YOUR CODE HERE
loss = history.history['loss'] ### YOUR CODE HERE
val_loss = history.history['val_loss'] ### YOUR CODE HERE
#-------------------------------------------------------
#predicted = np.array(model.predict(X)) # Unccomment when you use normalizer
#y_testtrue= np.array(y)  # Unccomment when you use normalizer
predicted = np.array(scaler2.inverse_transform(model.predict(X))) # Comment when you use normalizer
y_testtrue= np.array(scaler2.inverse_transform(y)) # Comment when you use normalizer
Number = range(len(y))
#-------------------------------------------------------
epochs = range(len(acc))
#-------------------------------------------------------
if modee==2:
 plt.figure()
 plt.subplot(2,1,1)
 plt.plot(Number, predicted[:,0], 'o-', label='Predicted')
 plt.plot(Number, y_testtrue[:,0],'x-', label='Real Data')
 plt.title('Evaporation Rate - train data')
 plt.legend()
 #-------------------------------------------------------
 plt.subplot(2,1,2)
 plt.plot(Number, predicted[:,1], 'o-', label='Predicted')
 plt.plot(Number, y_testtrue[:,1],'x-', label='Real Data')
 plt.title('Temperature - train data')
 plt.legend()
#-------------------------------------------------------
elif modee==1:
 plt.figure()
 plt.plot(Number, predicted, 'o-', label='Predicted')
 plt.plot(Number, y_testtrue,'x-', label='Real Data')
 plt.title('Efficiency - train data')
 plt.legend()
#-------------------------------------------------------
#predicted = np.array(model.predict(X)) # Unccomment when you use normalizer
#y_testtrue= np.array(y) # Unccomment when you use normalizer
predicted = np.array(scaler2.inverse_transform(model.predict(X_test))) # Comment when you use normalizer
y_testtrue= np.array(scaler2.inverse_transform(y_test)) # Comment when you use normalizer
Number = range(len(y_test))
#-------------------------------------------------------
epochs = range(len(acc))
#-------------------------------------------------------
if modee==2:
 plt.figure()
 plt.subplot(2,1,1)
 plt.plot(Number, predicted[:,0], 'o-', label='Predicted')
 plt.plot(Number, y_testtrue[:,0],'x-', label='Real Data')
 plt.title('Evaporation Rate - test data')
 plt.legend()
 #-------------------------------------------------------
 plt.subplot(2,1,2)
 plt.plot(Number, predicted[:,1], 'o-', label='Predicted')
 plt.plot(Number, y_testtrue[:,1],'x-', label='Real Data')
 plt.title('Temperature - test data')
 plt.legend()
 #-------------------------------------------------------
elif modee==1:
 plt.figure()
 plt.plot(Number, predicted, 'o-', label='Predicted')
 plt.plot(Number, y_testtrue,'x-', label='Real Data')
 plt.title('Efficiency - test data')
 plt.legend()
#-------------------------------------------------------
#predicted = np.array(model.predict(X)) # Unccomment when you use normalizer
#y_testtrue= np.array(y) # Unccomment when you use normalizer
predicted = np.array(scaler2.inverse_transform(model.predict(X_val))) # Comment when you use normalizer
y_testtrue= np.array(scaler2.inverse_transform(y_val)) # Comment when you use normalizer
Number = range(len(y_val))
#-------------------------------------------------------
epochs = range(len(acc))
#-------------------------------------------------------
if modee==2:
 plt.figure()
 plt.subplot(2,1,1)
 plt.plot(Number, predicted[:,0], 'o-', label='Predicted')
 plt.plot(Number, y_testtrue[:,0],'x-', label='Real Data')
 plt.title('Evaporation Rate - validation data')
 plt.legend()
 #-------------------------------------------------------
 plt.subplot(2,1,2)
 plt.plot(Number, predicted[:,1], 'o-', label='Predicted')
 plt.plot(Number, y_testtrue[:,1],'x-', label='Real Data')
 plt.title('Temperature - validation data')
 plt.legend()
 #-------------------------------------------------------
elif modee==1:
 plt.figure()
 plt.plot(Number, predicted, 'o-', label='Predicted')
 plt.plot(Number, y_testtrue,'x-', label='Real Data')
 plt.title('Efficiency - validation data')
 plt.legend()
#-------------------------------------------------------
plt.show() 
#-------------------------------------------------------
#print('-------------------------------------------------------')
#pridicted = scaler.inverse_transform(model.predict(X_test))
#y_testtrue= scaler.inverse_transform(y_test)
#print(predicted)
#print(y_testtrue)
#print(Number)
#print('-------------------------------------------------------')
