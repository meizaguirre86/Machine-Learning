import pandas as pd
import quandl, math
import datetime
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

#    Once downloaded we store dataframe to googl.csv
#quandl.ApiConfig.api_key = "rRMS18tGdeyUeLY_vqkQ"
#df = quandl.get('WIKI/GOOGL')
#df.to_csv('googl.csv')
df = pd.read_csv('googl.csv', index_col='Date', parse_dates=True)

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Close'] * 100.0
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

#forecast_out = 25
forecast_out = int(math.ceil(0.1*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label', 'Adj. Close'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]


df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf_lr = LinearRegression(n_jobs=1)
clf_lr.fit(X_train,y_train)
clf_svm = svm.SVR(kernel='sigmoid')
clf_svm.fit(X_train,y_train)
clf_mlp = MLPRegressor(solver='lbfgs', alpha=10,learning_rate = 'invscaling', hidden_layer_sizes=(6,6), random_state=1, max_iter = 10000)
clf_mlp.fit(X_train,y_train)

accuracy_svm = clf_svm.score(X_test,y_test)
accuracy_lr = clf_lr.score(X_test,y_test)
accuracy_mlp = clf_mlp.score(X_test,y_test)
forecast_set_lr = clf_lr.predict(X_lately)
forecast_set_svm = clf_svm.predict(X_lately)
forecast_set_mlp = clf_mlp.predict(X_lately)

print(accuracy_lr, accuracy_svm, accuracy_mlp)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set_lr:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range (len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

