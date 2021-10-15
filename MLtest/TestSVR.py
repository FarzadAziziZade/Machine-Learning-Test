from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#-------------------------------------------------------
csv_file_name = "datain"
#-------------------------------------------------------
header_list = ["f1", "f2", "f3", "f4", "o5", "o6", "o7"]
data = pd.read_csv('{file_name}.csv'.format(file_name = csv_file_name), names=header_list).replace('?', np.NaN).dropna()
X = data.get(["f1", "f2", "f3", "f4"])
y = data.get(["o7"])
scaler = StandardScaler()
scaler1 = StandardScaler()
X = scaler1.fit_transform(X)
y = scaler.fit_transform(y)
#-------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle = True)
#-------------------------------------------------------
rng = np.random.RandomState(0)
# kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
regr = make_pipeline(StandardScaler(), SVR(kernel='linear',C=1, epsilon=0.1))
#regr = make_pipeline(StandardScaler(), SVR(kernel='sigmoid',C=1, epsilon=0.1))
#regr = make_pipeline(StandardScaler(), SVR(C=1, epsilon=0.1))
#-------------------------------------------------------
regr.fit(X_train, y_train)
#-------------------------------------------------------
print(regr.score(X_test, y_test, sample_weight=None))
#-------------------------------------------------------
import matplotlib.pyplot as plt
predicted = np.array(scaler.inverse_transform(regr.predict(X)))
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