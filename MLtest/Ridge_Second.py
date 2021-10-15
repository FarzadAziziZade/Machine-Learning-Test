#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

#Importing the dataset
csv_file_name = "datain"
#-------------------------------------------------------
header_list = ["f1", "f2", "f3", "f4", "o5", "o6", "o7"]
data = pd.read_csv('{file_name}.csv'.format(file_name = csv_file_name), names=header_list).replace('?', np.NaN).dropna()
X = data.get(["f1", "f2", "f3", "f4"])
y = data.get(["o7"])

#Feature Scaling
sc_X = StandardScaler()
scaler = StandardScaler()
X = sc_X.fit_transform(X)
y = scaler.fit_transform(y)

#fit model
model = Ridge(alpha=0.99)
model.fit(X,y)
print(model.score(X, y))


#plot
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