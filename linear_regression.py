#Import all libraries
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pandas.plotting import register_matplotlib_converters
from sklearn import linear_model
register_matplotlib_converters()
plt.style.use('bmh')
from scipy.optimize import curve_fit

merged_data = pd.read_csv('clean/final_merged_data.csv')

#### SCATTERPLOTS FOR LINEAR-EXPONENTIAL REGRESSION ####
#Scatterplot for linear regression: https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
#Youtube reference: https://www.youtube.com/watch?v=8jazNUpO3lQ&t=45s
#Youtube Reference Method #2: https://www.youtube.com/watch?v=MRm5sBfdBBQ&t=192s

just_tsla_nasdaq = merged_data[['Tesla Close Price','NASDAQ Close Price']]

X = np.array(just_tsla_nasdaq['NASDAQ Close Price'])
X = X.reshape(-1,1)
Y = np.array(just_tsla_nasdaq['Tesla Close Price'])

X_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.25)


tsla_reg = LinearRegression()
tsla_reg.fit(X_train, y_train)

#Manually seeing y = mx + b
#print(tsla_reg.predict([[7700]]))
#print(tsla_reg.coef_)
#print(tsla_reg.intercept_)
#sumthing = 0.03078528*9000 + -137.20525079367906


d = X
p = tsla_reg.predict(d)
just_tsla_nasdaq['Predicted'] = p
print(just_tsla_nasdaq)

plt.figure(figsize=(16,8))
plt.title('Linear Regression Tesla/NASDAQ')
plt.xlabel('NASDAQ Close Price in USD')
plt.ylabel('Tesla Close Price in USD')
plt.scatter(just_tsla_nasdaq['NASDAQ Close Price'], just_tsla_nasdaq['Tesla Close Price'], label = 'Actual Values')
plt.plot(X, just_tsla_nasdaq['Predicted'], color = 'red', label = 'Regression Line')
plt.legend()
plt.savefig('graphs/Tesla_NASDAQ_Linear_Regression.ps')

plt.show()

