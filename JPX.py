import numpy as np
import pandas as pd
import mplfinance
from pandas_datareader import data
import matplotlib.pyplot as plt
import datetime
import plotly.graph_objects as go
import math
from sklearn import metrics

# Simple Linear Regression
from sklearn.linear_model import LinearRegression


# Multiple Linear Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score
from mpl_toolkits import mplot3d
from sklearn.model_selection import TimeSeriesSplit


# Import the necessary packages for K-Means Clustering
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans

path = "dataset/train_files/stock_prices.csv"
df = pd.read_csv(path)


print(df.size)
# df = df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
df = df[df['High'].notna()]
print(df.size)


# print(df.head)
# print(df.columns)
# print(df["Target"])
# df.index = df["Date"]
# print(df.iloc[0, :])
# print(df.count)

list_of_companies = [
    {
        'Name': 'Astellas Pharma Inc.',
        'SecuritiesCode': 4503,
        'Industry': 'Pharmaceutical'
    },
    {
        'Name': 'Canon Electronics Inc.',
        'SecuritiesCode': 7739,
        'Industry': 'Electric Appliances'

    },
    {
        'Name': 'Honda Motor Co. Ltd. ',
        'SecuritiesCode': 7267,
        'Industry': 'Transportation Equipment'
    },
    {
        'Name': 'Hitachi Ltd.',
        'SecuritiesCode': 6501,
        'Industry': 'Electric Appliances'
    },
    {
        'Name': 'SoftBank Corp.',
        'SecuritiesCode': 9434,
        'Industry': 'Information and Communication'
    },
    {
        'Name': 'Mitsubishi Motors Corporation',
        'SecuritiesCode': 7211,
        'Industry': 'Transportation Equipment'
    },
    {
        'Name': 'Nissan Motors Corp',
        'SecuritiesCode': 7201,
        'Industry': 'Transportation Equipment'
    },
    {
        'Name': 'Toyota Motor Corp',
        'SecuritiesCode': 7203,
        'Industry': 'Transportation Equipment'
    },
    {
        'Name': 'Sony Group Corporation',
        'SecuritiesCode': 6758,
        'Industry': 'Electric Appliances'
    },
    {
        'Name': 'Panasonic Corporation',
        'SecuritiesCode': 6752,
        'Industry': 'Electric Appliances'
    }
]

# for x in list_of_companies:
#     print("------")
#     print("Name: " + str(x['Name']) + "(" + str(x['SecuritiesCode']) + ")")
#     print("Industry: " + str(x['Industry']))


def fetchCompanyData(code, df):
    df2 = df.loc[df['SecuritiesCode'] == code]
    return df2


def plotData(df):
    df2 = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df2.index = pd.DatetimeIndex(df2['Date'])
    mplfinance.plot(df2)


def multipleLinearRegression(df):
    data = df[['Open', 'High', 'Low', 'Volume', 'Close']]
    print(data.head())

    # X = np.column_stack(
    #     [np.ones(len(df['x'])), df["x"].values.reshape(1, -1)[0]])

    data_X = data.loc[:, data.columns != 'Close']
    # data_X = np.column_stack(
    #     [np.ones(len(data_X['Open'])), data_X.values.reshape(1, -1)[0]])

    test2 = data_X["Open"].values.reshape(1, -1)[0]
    test1 = np.ones(len(data_X['Open']))
    data_X = np.column_stack(
        [test1, test2, data_X['High'].values.reshape(1, -1)[0], data_X['Low'].values.reshape(1, -1)[0], data_X['Volume'].values.reshape(1, -1)[0]])

    print(test1.shape)
    print(test2.shape)

    data_Y = data['Close']
    # data_Y = data_Y.values.reshape(1, -1)

    print(data_X.shape)
    # print(data_X)
    print(data_Y.shape)
    # print(data_Y)

    train_X, test_X, train_y, test_y = train_test_split(
        data_X, data_Y, test_size=0.5)

    print('\n\nTraining Set')

    # Creating the Regressor
    regressor = LinearRegression()
    regressor.fit(train_X, train_y)

    # Make Predictions and Evaluate the results
    predict_y = regressor.predict(test_X)
    print('Prediction Score : ', regressor.score(test_X, test_y))
    error = mean_squared_error(test_y, predict_y)
    print('Mean Squared Error : ', error)

    coeff = regressor.coef_
    print("Coefficients: ", coeff)

    # Plot the predicted and the expected values
    # fig = plt.figure()
    # ax = plt.axes()
    # ax.grid()
    # ax.set(xlabel='Close ($)', ylabel='Open ($)',
    #        title='Panasonic Multiple Regression')
    # ax.plot(test_X['Open'], test_y)
    # ax.plot(test_X['Open'], predict_y)
    # fig.savefig('LRPlot.png')
    # plt.show()

    return coeff, predict_y


def simpleLinearRegression(df):
    dates = []
    for x in range(0, len(df["Date"])):
        dates.append(x)
    prices = df['Close']

    dates = np.asanyarray(dates)
    prices = np.asanyarray(prices)
    dates = np.reshape(dates, (len(dates), 1))
    prices = np.reshape(prices, (len(prices), 1))

    xtrain, xtest, ytrain, ytest = train_test_split(
        dates, prices, test_size=0.2)
    # best = reg.score(ytrain, ytest)
    best = 0
    # bestReg
    for _ in range(100):
        xtrain, xtest, ytrain, ytest = train_test_split(
            dates, prices, test_size=0.2)
        reg = LinearRegression().fit(xtrain, ytrain)
        acc = reg.score(xtest, ytest)
        if acc > best:
            best = acc
            bestReg = reg

    mean = 0
    for i in range(10):
        msk = np.random.rand(len(df)) < 0.8
        xtest = dates[~msk]
        ytest = prices[~msk]
        mean += bestReg.score(xtest, ytest)

    print("Average Accuracy: ", mean/10)

    # Plot Predicted VS Actual Data
    plt.plot(xtest, ytest, color='green', linewidth=1,
             label='Actual Price')  # plotting the initial datapoints
    plt.plot(xtest, bestReg.predict(xtest), color='blue', linewidth=3,
             label='Predicted Price')  # plotting the line made by linear regression
    plt.title('Linear Regression | Time vs. Price ')
    plt.legend()
    plt.xlabel('Date Integer')
    plt.show()


def plotMultipleRegressionLine(df, coeff):
    date_close = df[['Date', 'Open', 'High', 'Low', 'Volume', 'Close']]
    date_close.index = date_close['Date']
    plt.xlabel("date")
    plt.ylabel("$ price")

    print("Coefficients: ", coeff)
    plt.plot(date_close.index, date_close['Close'])
    plt.plot(date_close.index, (coeff[0] * 0 + coeff[1] * date_close['Open'] +
             coeff[2] * date_close['High'] + coeff[3] * date_close['Low'] + coeff[4] * date_close['Volume']), '-')
    ax = plt.axes()
    ax.grid()

    plt.show()


def testLinearRegression(df):
    dataset = df[['Open', 'Low', 'Close']]
    print(dataset['Open'].shape)
    print(dataset['Close'].shape)
    model1 = LinearRegression()

    # X = dataset['Open'].values.reshape(1, -1)[0]

    X = np.column_stack(
        [np.ones(len(dataset['Open'])), dataset["Open"].values.reshape(1, -1)[0], dataset["Low"].values.reshape(1, -1)[0]])
    y = dataset["Close"]
    model1.fit(X, y)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # ax.plot3D(xline, yline, zline, 'gray')

    xdata = dataset['Open']
    ydata = dataset['Low']
    zdata = dataset['Close']
    ax.scatter3D(xdata, ydata, zdata, cmap='Greens')
    # plt.plot(dataset['Open'].values, dataset['Close'].values, '.')
    # plt.plot(dataset['Open'].values, model1.predict(X), '-')
    plt.show()


# Fetch data for Panasonic
df_6752 = fetchCompanyData(6752, df)
# plotData(df_6752)

# Multiple Linear Regression
# coeff, predict_y = multipleLinearRegression(df_6752)

# plotMultipleRegressionLine(df_6752, coeff, )

# testLinearRegression(df_6752)
simpleLinearRegression(df_6752)


# Plot data
# plotData(df_6752)
# plotData(df_6752)


# Selecting Multiple Security Code
# mask0 = df['SecuritiesCode'] == 1332
# mask1 = df['SecuritiesCode'] == 1333
# masks = mask0 | mask1
# targets = df.loc[masks]
# print(targets)
