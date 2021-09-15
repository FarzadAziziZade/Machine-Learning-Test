# Farzad Azizi Zade
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
y = data.get(["o5", "o6"])
#-------------------------------------------------------
#data.head() # this function shows only the first 5 values
#-------------------------------------------------------
# initialising the Scaler
scaler = MinMaxScaler()
scaler2 = MinMaxScaler()
#scaler = minmax_scale()
#scaler = StandardScaler()
#scaler=Normalizer() # it is the best one, but it does not have inverse
# learning the statistical parameters for each of the data and transforming
X = scaler.fit_transform(X)
y = scaler2.fit_transform(y)
#-------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle = True)
print(X_train)
print(X_test)
print(X_val)
print(y_train)
print(y_test)
print(y_val)
#-------------------------------------------------------
import tensorflow as tf
#-------------------------------------------------------
model = tf.keras.models.Sequential([                 
    tf.keras.layers.Dense(8, activation='tanh', input_dim = 4, name="Dense_1"),
    tf.keras.layers.Dense(2, activation='tanh', name="Output")
], name="Model_1")
model.compile(  optimizer='RMSprop', 
                loss='mean_squared_error', 
                metrics=['accuracy'])
#model.summary() #it will print test data and predicted results
#-------------------------------------------------------
history = model.fit(  X_train, 
            y_train, 
            epochs=1000,
            validation_data=(X_val, y_val),
            verbose=1)
#-------------------------------------------------------
model.evaluate(X_test, y_test, verbose=1)
#-------------------------------------------------------
print(model.predict(X_test))
print(y_test)
#-------------------------------------------------------
model.save("Test.h5")
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
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
#-------------------------------------------------------
plt.figure()
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()
#-------------------------------------------------------
plt.figure()
plt.plot(Number, predicted[:,0], 'r', label='Predicted')
plt.plot(Number, y_testtrue[:,0], 'b', label='Real Data')
plt.title('Evaporation Rate')
plt.legend()
#-------------------------------------------------------
plt.figure()
plt.plot(Number, predicted[:,1], 'r', label='Predicted')
plt.plot(Number, y_testtrue[:,1], 'b', label='Real Data')
plt.title('Temperature')
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
