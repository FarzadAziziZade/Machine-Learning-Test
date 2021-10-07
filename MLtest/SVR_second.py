#1 Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2 Importing the dataset
csv_file_name = "datain"
#-------------------------------------------------------
header_list = ["f1", "f2", "f3", "f4", "o5", "o6", "o7"]
data = pd.read_csv('{file_name}.csv'.format(file_name = csv_file_name), names=header_list).replace('?', np.NaN).dropna()
X = data.get(["f1", "f2", "f3", "f4"])
y = data.get(["o7"])

#3 Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

#4 Fitting the Support Vector Regression Model to the dataset
# Create your support vector regressor here
from sklearn.svm import SVR
# most important SVR parameter is Kernel type. It can be 
# linear,polynomial or gaussian SVR.
# We have a non-linear condition #so we can select polynomial or
# gaussian but here we select RBF(a #gaussian type) kernel.
regressor = SVR(kernel='poly', degree=21)
regressor.fit(X,y)

#5 Predicting a new result
y_pred = regressor.predict(X)

#6 Visualising the Support Vector Regression results
import matplotlib.pyplot as plt
predicted = regressor.predict(X)
y_testtrue= y
Number = range(len(y))
print(regressor.score(X, y, sample_weight=None))
#-------------------------------------------------------
plt.figure()
plt.plot(Number, predicted, 'r', label='Predicted')
plt.plot(Number, y_testtrue, 'b', label='Real Data')
plt.title('Efficiency')
plt.legend()
#-------------------------------------------------------
plt.show() 