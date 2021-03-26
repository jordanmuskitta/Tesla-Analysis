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
register_matplotlib_converters()
import pandas_datareader as pdr
from scipy.optimize import curve_fit



#This is to change from sceintific notation to real numbers, reference: https://stackoverflow.com/questions/55394854/how-to-change-the-format-of-describe-output
pd.set_option('display.float_format', lambda x: '%.3f' % x)

nasdaq = pd.read_csv("NASDAQ.csv")
lit = pd.read_csv("LIT.csv")
nickel = pd.read_csv("NICKEL.csv")
tsla = pd.read_csv("TSLA.csv")

#Set Index to Date Columns:
nickel = nickel.set_index('Date')
nasdaq = nasdaq.set_index('Date')
lit = lit.set_index('Date')
tsla = tsla.set_index('Date')


#FOR NICKEL DATASET
#Adjust Start and End Dates:

nickel = nickel.reindex(index=nickel.index[::-1])

nickel_date_set = nickel.loc['2015-09-24':'2020-09-24']

#Clean Dataset (Zero Values or Null Values):
new_vol_lst = []
for x in nickel_date_set:
  if x == 'Vol.':
    for i in nickel_date_set[x]:
      new_Vol = i.rstrip('K')
      if i == '-':
        new_Vol = new_Vol.replace('-', '0.00')
        new_vol_lst.append(float(new_Vol))
      else:
        float_Vol = float(new_Vol)
        new_vol_lst.append(float(new_Vol))

  else:
    nickel_date_set[x] = nickel_date_set[x].str.replace(',', '')
    nickel_date_set[x] = nickel_date_set[x].str.replace('%', '')
    nickel_date_set[x] = pd.to_numeric(nickel_date_set[x], downcast="float")

#Convert the to whole numbers (*1000):
new_vol_numeric = []
for values in new_vol_lst:
  new_vol_numeric.append(values*1000)

#Add new column into the data set:
nickel_date_set['New Vol.'] = new_vol_numeric

#Pandas to CSV, this creates a csv file at path.
nickel_date_set.to_csv('clean/new_nickel.csv')

#For NASDAQ Dataset:

#Adjust Start and End Dates:
nasdaq = nasdaq.reindex(index=nasdaq.index[::-1])
nasdaq = nasdaq.reindex(index=nasdaq.index[::-1])
nasdaq_date_set = nasdaq[(nasdaq.index > '2015-09-24') & (nasdaq.index <= '2020-09-24')]

#Pandas to CSV, this creates a csv file at path.
nasdaq_date_set.to_csv('clean/new_nasdaq.csv')

#For Lithium ETF Dataset:

#Adjust Start and End Dates:
lit = lit.reindex(index=lit.index[::-1])
lit = lit.reindex(index=lit.index[::-1])
lit_date_set = lit[(lit.index > '2015-09-24') & (lit.index <= '2020-09-24')]

#Pandas to CSV, this creates a csv file at path.
lit_date_set.to_csv('clean/new_lit.csv')

#For TSLA Dataset:

#Adjust Start and End Dates:
tsla = tsla.reindex(index=tsla.index[::-1])
tsla = tsla.reindex(index=tsla.index[::-1])
tsla_date_set = tsla[(tsla.index > '2015-09-24') & (tsla.index <= '2020-09-24')]

#Pandas to CSV, this creates a csv file at path.
tsla_date_set.to_csv('clean/new_tsla.csv')

# Merge the DataFrames into one:

merged_data = pd.DataFrame()

#For Tesla:

merged_data['Tesla Close Price'] = tsla_date_set['Close']
merged_data['Tesla Volume'] = tsla_date_set['Volume']

#For Nickel:

merged_data['Nickel Close Price'] = nickel_date_set['Price']
merged_data['Nickel Volume'] = nickel_date_set['New Vol.']

#For Lithium ETF:

merged_data['Lithium Close Price'] = lit_date_set['Close']
merged_data['Lithium Volume'] = lit_date_set['Volume']

#For NASDAQ:

merged_data['NASDAQ Close Price'] = nasdaq_date_set['Close']
merged_data['NASDAQ Volume'] = nasdaq_date_set['Volume']

#Pandas to CSV, this creates a csv file at path.
merged_data.to_csv('clean/new_merged_data.csv')

new_merged_data = pd.read_csv('clean/new_merged_data.csv')

#Add day of the week into the dataframe = changing into series

merged_data['Day of the Week'] = pd.to_datetime(merged_data.index)
merged_data['Day of the Week'] = merged_data['Day of the Week'].dt.day_name()

#Add Moving Average:
merged_data['Tesla Price MA10'] = merged_data['Tesla Close Price'].rolling(10).mean()
merged_data['Tesla Price MA50'] = merged_data['Tesla Close Price'].rolling(50).mean()
merged_data['Lithium Price MA10'] = merged_data['Lithium Close Price'].rolling(10).mean()
merged_data['Lithium Price MA50'] = merged_data['Lithium Close Price'].rolling(50).mean()
merged_data['Nickel Price MA10'] = merged_data['Nickel Close Price'].rolling(10).mean()
merged_data['Nickel Price MA50'] = merged_data['Nickel Close Price'].rolling(50).mean()

#Percentage Change for each asset

merged_data['Percentage Change Tesla'] = merged_data['Tesla Close Price'].pct_change()
merged_data['Percentage Change NASDAQ'] = merged_data['NASDAQ Close Price'].pct_change()

merged_data = merged_data.fillna(method='bfill')

#Create Boolean filter for days of the week:

is_monday = merged_data['Day of the Week'] == 'Monday'
merged_monday = merged_data[is_monday]
merged_monday['Total Volume Tesla & NASDAQ'] = sum(merged_monday['Tesla Volume'], merged_monday['NASDAQ Volume'])
merged_monday.to_csv('clean/monday_merged_data.csv')

is_tuesday = merged_data['Day of the Week'] == 'Tuesday'
merged_tuesday = merged_data[is_tuesday]
merged_tuesday['Total Volume Tesla & NASDAQ'] = sum(merged_tuesday['Tesla Volume'], merged_tuesday['NASDAQ Volume'])
merged_tuesday.to_csv('clean/tuesday_merged_data.csv')

is_wednesday = merged_data['Day of the Week'] == 'Wednesday'
merged_wednesday = merged_data[is_wednesday]
merged_wednesday['Total Volume Tesla & NASDAQ'] = sum(merged_wednesday['Tesla Volume'], merged_wednesday['NASDAQ Volume'])
merged_wednesday.to_csv('clean/wednesday_merged_data.csv')

is_thursday = merged_data['Day of the Week'] == 'Thursday'
merged_thursday = merged_data[is_thursday]
merged_thursday['Total Volume Tesla & NASDAQ'] = sum(merged_thursday['Tesla Volume'], merged_thursday['NASDAQ Volume'])
merged_thursday.to_csv('clean/thursday_merged_data.csv')

is_friday = merged_data['Day of the Week'] == 'Friday'
merged_friday = merged_data[is_friday]
merged_friday['Total Volume Tesla & NASDAQ'] = sum(merged_friday['Tesla Volume'], merged_friday['NASDAQ Volume'])
merged_friday.to_csv('clean/friday_merged_data.csv')

merged_data.to_csv('clean/final_merged_data.csv')


#Count the number of positive returns according to day of the week

trading_dict = {}
trading_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
is_first_line = True
for line in open('clean/final_merged_data.csv'):
  if is_first_line is True:
    is_first_line = False
  else:
    line_strip = line.strip()
    line_split = line_strip.split(',')
    pct_change = line_split[16]
    day_week = line_split[9]
    if day_week not in trading_dict and '-' not in pct_change:
      trading_dict[day_week] = [pct_change]
    if day_week in trading_dict and '-' not in pct_change:
      trading_dict[day_week].append(pct_change)


num_mon = len(trading_dict['Monday'])
num_tue = len(trading_dict['Tuesday'])
num_wed =len(trading_dict['Wednesday'])
num_thur = len(trading_dict['Thursday'])
num_fri = len(trading_dict['Friday'])

print('\n')
print("Number of Positive Trading Days on Monday:", num_mon)
print("Number of Positive Trading Days on Tuesday:", num_tue)
print("Number of Positive Trading Days on Wednesday:", num_wed)
print("Number of Positive Trading Days on Thursday:", num_thur)
print("Number of Positive Trading Days on Friday:", num_fri)
print('\n')

#Aggregating price and volume values by days of the week



#### CHARTING/GRAPHING SECTION ####

#Correlation Coefficient:

merged_data_correlation = new_merged_data.corr(method = 'pearson')
merged_data_correlation.to_csv('clean/merged_data_correlation.csv')
print(merged_data_correlation)

merged_data['Tesla Close Price'].plot( legend = 'Price', figsize = (20,10)), merged_data['NASDAQ Close Price'].plot(legend = 'Price', figsize = (20,10))


#Merged Tesla and NASDAQ Price Plots:
plt.figure()
x_axis = new_merged_data['Date']
y1 = merged_data['Tesla Close Price']
y2 = merged_data['NASDAQ Close Price']
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
curve1 = ax1.plot(x_axis, y1, label = 'Tesla Close Price')
curve2 = ax2.plot(x_axis, y2, label = 'NASDAQ Close Price', color = 'r')
ax1.legend()
ax2.legend(loc = 'upper right')
ax1.set_xlabel('Date (250 Days)')
ax1.set_ylabel('Tesla Close Price ($USD)')
ax2.set_ylabel('NASDAQ Close Price ($USD)')
plt.xticks(x_axis[::250])
plt.savefig('Tesla_Nasdaq_Price.ps')

#Merged Lithium and Tesla:

plt.figure()
x_axis = new_merged_data['Date']
y1 = merged_data['Tesla Close Price']
y2 = merged_data['Lithium Close Price']
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
curve1 = ax1.plot(x_axis, y1, label = 'Tesla Close Price')
curve2 = ax2.plot(x_axis, y2, label = 'Lithium Close Price', color = 'r')
ax1.legend()
ax2.legend(loc = 'upper right')
ax1.set_xlabel('Date (250 Days)')
ax1.set_ylabel('Tesla Close Price ($USD)')
ax2.set_ylabel('Lithium Close Price ($USD)')
plt.xticks(x_axis[::250])
plt.title('Tesla Close Price and Lithium ETF Price in USD', loc= 'left')
plt.savefig('Tesla_Lithium_Price.ps')

#Merged Nickel and Tesla:

plt.figure()
x_axis = new_merged_data['Date']
y1 = merged_data['Tesla Close Price']
y2 = merged_data['Nickel Close Price']
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
curve1 = ax1.plot(x_axis, y1, label = 'Tesla Close Price')
curve2 = ax2.plot(x_axis, y2, label = 'Nickel Close Price', color = 'r')
ax1.legend()
ax2.legend(loc = 'upper right')
ax1.set_xlabel('Date (250 Days)')
ax1.set_ylabel('Tesla Close Price ($USD)')
ax2.set_ylabel('Nickel Close Price ($USD)')
plt.title('Tesla Close Price and Nickel Spot Price in USD', loc= 'left')
plt.xticks(x_axis[::250])
plt.savefig('Tesla_Nickel_Price.ps')

#Scatterplot for Tesla and NASDAQ Price and Volume, grouped in days
#Reference for color bar: https://stackoverflow.com/questions/46106912/one-colorbar-for-multiple-scatter-plots
#Reference for Z-values: https://stackoverflow.com/questions/64735131/multiple-scatter-plots-and-one-colorbar

plt.figure()
plt.style.use('seaborn')
fig, ax = plt.subplots()
z1 = merged_monday['Total Volume Tesla & NASDAQ']
z2 = merged_tuesday['Total Volume Tesla & NASDAQ']
z3 = merged_wednesday['Total Volume Tesla & NASDAQ']
z4 = merged_thursday['Total Volume Tesla & NASDAQ']
z5 = merged_friday['Total Volume Tesla & NASDAQ']
mini, maxi = 0, 2
norm = plt.Normalize(mini, maxi)
zs = np.concatenate([z1, z2, z3, z4, z5], axis=0)
min_, max_ = zs.min(), zs.max()


mon_scat = plt.scatter(merged_monday['Percentage Change NASDAQ'], merged_monday['Percentage Change Tesla'], c=z1,cmap='viridis_r', norm=norm,marker= 'o', label = 'Monday', edgecolors = 'black')
plt.clim(min_, max_)
tue_scat = plt.scatter(merged_tuesday['Percentage Change NASDAQ'], merged_tuesday['Percentage Change Tesla'],c=z2,cmap='viridis_r', marker= 'x', label = 'Tuesday')
plt.clim(min_, max_)
wed_scat = plt.scatter(merged_wednesday['Percentage Change NASDAQ'], merged_wednesday['Percentage Change Tesla'],c=z3,cmap='viridis_r',marker= 'v', label = 'Wednesday', edgecolors = 'black')
plt.clim(min_, max_)
thurs_scat = plt.scatter(merged_thursday['Percentage Change NASDAQ'], merged_thursday['Percentage Change Tesla'],c=z4,cmap='viridis_r',marker= 'd', label = 'Thursday', edgecolors = 'black')
plt.clim(min_, max_)
fri_scat = plt.scatter(merged_friday['Percentage Change NASDAQ'], merged_friday['Percentage Change Tesla'],c=z5,cmap='viridis_r',marker= 's', label = 'Friday', edgecolors = 'black')
plt.clim(min_, max_)
plt.colorbar(orientation = 'horizontal').set_label('Combined Volume Traded (in tens of billions, * 1e10)')
plt.title('Tesla and NASDAQ, Volume and Price Grouped in Days ', loc= 'left')
ax.set_xlabel("Percentage Change NASDAQ (Decimal)")
ax.set_ylabel("Percentage Change Tesla (Decimal)")
plt.axvline(0, c = (0.5, 0.5, 0.5), ls= '--')
plt.axhline(0, c = (0.5, 0.5, 0.5), ls= '--')
ax.legend()
plt.savefig('Tesla_NASDAQ_Scatter.ps')


#Scatterplot for broken up into individual days:
#Reference: https://riptutorial.com/matplotlib/example/11257/grid-of-subplots-using-subplot

plt.figure()

plt.style.use('seaborn')
fig, axes = plt.subplots(2, 3,constrained_layout=True, figsize=(10, 5),sharey=True, sharex=True)
#mini, maxi = 0, 2
#norm = plt.Normalize(mini, maxi)
#zs = np.concatenate([z1, z2, z3, z4, z5], axis=0)
#min_, max_ = zs.min(), zs.max()
fig.text(0.5,0.04, "Percentage Change in NASDAQ (Decimal Form)", ha="center", va="center")
fig.text(0.05,0.5, "Percentage Change in Tesla (Decimal Form)", ha="center", va="center", rotation=90)

mon_ax = axes[0,0].scatter(merged_monday['Percentage Change NASDAQ'], merged_monday['Percentage Change Tesla'],c=z1,cmap='viridis_r', norm=norm,marker= 'o', label = 'Monday', edgecolors = 'black')
axes[0,0].set_title("Monday Returns and Volume")
#plt.clim(min_, max_)

tue_ax = axes[0,1].scatter(merged_tuesday['Percentage Change NASDAQ'], merged_tuesday['Percentage Change Tesla'],c=z2,cmap='viridis_r', norm=norm,marker= 'o', label = 'Tuesday', edgecolors = 'black')
axes[0,1].set_title("Tuesday Returns and Volume")
#plt.clim(min_, max_)

wed_ax = axes[0,2].scatter(merged_wednesday['Percentage Change NASDAQ'], merged_wednesday['Percentage Change Tesla'],c=z3,cmap='viridis_r', norm=norm,marker= 'o', label = 'Wednesday', edgecolors = 'black')
axes[0,2].set_title("Wednesday Returns and Volume")
#plt.clim(min_, max_)

thurs_ax = axes[1,0].scatter(merged_thursday['Percentage Change NASDAQ'], merged_thursday['Percentage Change Tesla'],c=z4,cmap='viridis_r', norm=norm,marker= 'o', label = 'Thursday', edgecolors = 'black')
axes[1,0].set_title("Thursday Returns and Volume")
#plt.clim(min_, max_)

fri_ax = axes[1,1].scatter(merged_friday['Percentage Change NASDAQ'], merged_friday['Percentage Change Tesla'],c=z5,cmap='viridis_r', norm=norm,marker= 'o', label = 'Thursday', edgecolors = 'black')
axes[1,1].set_title("Friday Returns and Volume")
#plt.clim(min_, max_)

fig.delaxes(axes[1][2])
fig.suptitle('Returns and Volume in Days of the Week')


plt.savefig('Tesla_NASDAQ_Scatter_Seperated.ps')


#### LSTM Prediction ####
#Reference: https://www.datacamp.com/community/tutorials/lstm-python-stock-market


#### Decision Tree ####




#### SCATTERPLOTS FOR LINEAR-EXPONENTIAL REGRESSION ####
#Scatterplot for linear regression: https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html












'''

merged_scatter_tsla_nasdaq = merged_data.plot(kind = 'scatter', x='NASDAQ Close Price', y = 'Tesla Close Price')
merged_scatter_tsla_lithium = merged_data.plot(kind = 'scatter',  x = 'Lithium Close Price', y = 'Tesla Close Price')
merged_scatter_tsla_nickel = merged_data.plot(kind = 'scatter',  x = 'Nickel Close Price', y = 'Tesla Close Price')
merged_scatter_nasdaq_nickel = merged_data.plot(kind = 'scatter',  x = 'Nickel Close Price', y = 'NASDAQ Close Price')

plt.figure()
initial_guess = [1.0,1.0]
x1 = merged_data['NASDAQ Close Price'].values
x1 = np.array(x1)
print(type(x1))
y1 = merged_data['Tesla Close Price'].values
y1 = np.array(y1)
plt.plot(x1, y1, 'bo' ,label='Tesla NASDAQ Exponential Regression')
ans, cov = curve_fit(func, x1, y1, initial_guess)
fit_h0, fit_v0 = ans
t = np.linspace(0,2)
plt.plot(t, func(t,fit_h0, fit_v0),'r', label = 'model')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('Tesla_NASDAQ_Regression.ps')'''


plt.show()

total_dataset = pd.read_csv('clean/final_merged_data.csv')
print(total_dataset.describe())