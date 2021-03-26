#Import all libraries
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
plt.style.use('bmh')



#IMPORT DATASET:
#Reference: https://www.youtube.com/watch?v=hOLSGMEEwlI

merged_data = pd.read_csv('clean/final_merged_data.csv')
merged_data_two = merged_data[['Tesla Close Price']]
print(merged_data_two)
plt.plot(merged_data_two)
#Prepare dataset for test and train:

future_days = 60

merged_data_two['Prediction'] = merged_data_two['Tesla Close Price'].shift(-future_days)
X = np.array(merged_data_two.drop(['Prediction'], 1))[:-future_days]
#print(X)
Y = np.array(merged_data_two['Prediction'])[:-future_days]
#print(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25)
tree = DecisionTreeRegressor().fit(x_train, y_train)
lr = LinearRegression().fit(x_train, y_train)

x_future = merged_data_two.drop(['Prediction'], 1)[:-future_days]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)

tree_prediction = tree.predict(x_future)
lr_prediction = lr.predict(x_future)

predictions = tree_prediction
valid = merged_data_two[X.shape[0]:]
valid['Predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Days')
plt.ylabel('Price in AUD')
plt.plot(merged_data['Tesla Close Price'])
plt.plot(valid[['Tesla Close Price', 'Predictions']])
plt.legend(['Orig', 'Val', 'Pred'])
plt.plot(valid)
plt.savefig('graphs/Tesla_DecisionTree.ps')
plt.show()

predictions = lr_prediction
valid = merged_data_two[X.shape[0]:]
valid['Predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Days')
plt.ylabel('Price in AUD')
plt.plot(merged_data['Tesla Close Price'])
plt.plot(valid[['Tesla Close Price']], c = 'red')
plt.plot(valid[['Predictions']], c = 'green')
plt.legend(['Orig', 'Val', 'Pred'])
plt.plot(valid)
plt.savefig('graphs/Tesla_DecisionTree2.ps')
plt.show()

