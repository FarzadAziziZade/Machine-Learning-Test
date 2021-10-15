# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor
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
#csv_file_name = "datainn"
#-------------------------------------------------------
header_list = ["f1", "f2", "f3", "f4", "o5", "o6", "o7"]
data = pd.read_csv('{file_name}.csv'.format(file_name = csv_file_name), names=header_list).replace('?', np.NaN).dropna()
X = data.get(["f1", "f2", "f3", "f4"])
y = data.get(["o5", "o6"])
#-------------------------------------------------------
#data.head() # this function shows only the first 5 values
#-------------------------------------------------------
# initialising the Scaler
scaler= MinMaxScaler()
#scaler = minmax_scale()
#scaler = StandardScaler()
#scaler=Normalizer() # it is the best one, but it does not have inverse
# learning the statistical parameters for each of the data and transforming
scaler1=scaler
scaler2=scaler
X = scaler1.fit_transform(X)
y = scaler2.fit_transform(y)
#-------------------------------------------------------
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1,shuffle=True)
#model = MLPRegressor(hidden_layer_sizes=(1),activation='tanh',solver='lbfgs',random_state=1, verbose=0)
#model.fit(X_train, y_train)
#print(model.score(X_test, y_test))
#-------------------------------------------------------
from numpy import arange
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
#define model
estim = MLPRegressor()

# define model evaluation method
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)

# define grid
grid = {'hidden_layer_sizes':(1,2,3,4,7),'activation':('tanh','relu','logistic','identity'),'solver':('lbfgs','sgd','adam')}

# define search
# https://scikit-learn.org/stable/modules/model_evaluation.html
search = GridSearchCV(estim, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
#search = GridSearchCV(estim, grid, cv=cv, n_jobs=-1)

# perform the search
results = search.fit(X, y)
# summarize
print('MAE: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)



#-------------------------------------------------------
#import matplotlib.pyplot as plt
#predicted = np.array(scaler2.inverse_transform(model.predict(X)))
#y_testtrue= np.array(scaler2.inverse_transform(y))
#Number = range(len(y))
#-------------------------------------------------------
#plt.figure()
#plt.plot(Number, predicted[:,0], 'r', label='Predicted')
#plt.plot(Number, y_testtrue[:,0],'x', label='Real Data')
#plt.title('Evaporation Rate')
#plt.legend()
#-------------------------------------------------------
#plt.figure()
#plt.plot(Number, predicted[:,1], 'r', label='Predicted')
#plt.plot(Number, y_testtrue[:,1],'x', label='Real Data')
#plt.title('Temperature')
#plt.legend()
#-------------------------------------------------------
#plt.show()