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
scaler = StandardScaler()
X = sc_X.fit_transform(X)
y = scaler.fit_transform(y)

#4 Fitting the Support Vector Regression Model to the dataset
# Create your support vector regressor here


# evaluate an ridge regression model on the dataset
from numpy import mean
from numpy import std
from numpy import absolute
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge

# define model
model = Ridge(alpha=1.0)
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1)
# evaluate model
#scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
scores = cross_val_score(model, X, y, cv=cv, n_jobs=-1)
# force scores to be positive
#scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))


# grid search hyperparameters for ridge regression
from numpy import arange
from pandas import read_csv
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge

# define model
model = Ridge()
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid
grid = dict()
grid['alpha'] = arange(0, 1, 0.01)
# define search
#search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
search = GridSearchCV(model, grid, cv=cv, n_jobs=-1)
# perform the search
results = search.fit(X, y)
# summarize
print('MAE: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)

model.fit(X,y)
#-------------------------------------------------------
import matplotlib.pyplot as plt
predicted = np.array(scaler.inverse_transform(model.predict(X)))
y_testtrue= np.array(scaler.inverse_transform(y))
Number = range(len(y))
#-------------------------------------------------------
plt.figure()
plt.plot(Number, predicted, 'r', label='Predicted')
plt.plot(Number, y_testtrue,'x-', label='Real Data')
plt.title('Efficiency')
plt.legend()
#-------------------------------------------------------
plt.show() 