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

# Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures

# SVR
from sklearn.svm import SVR
import matplotlib.dates as mdates
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from matplotlib.colors import Normalize


# LSTM
# import tensorflow
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import LSTM
# from tensorflow.keras.layers import Dropout
# from tensorflow.keras.layers import *
# from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error


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
from sqlalchemy import false

path = "dataset/train_files/stock_prices.csv"
df = pd.read_csv(path)


# print(df.size)
# df = df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
df = df[df['High'].notna()]
# print(df.size)


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


# def multipleLinearRegression(df):
#     data = df[['Open', 'High', 'Low', 'Volume', 'Close']]
#     print(data.head())

#     # X = np.column_stack(
#     #     [np.ones(len(df['x'])), df["x"].values.reshape(1, -1)[0]])

#     data_X = data.loc[:, data.columns != 'Close']
#     # data_X = np.column_stack(
#     #     [np.ones(len(data_X['Open'])), data_X.values.reshape(1, -1)[0]])

#     test2 = data_X["Open"].values.reshape(1, -1)[0]
#     test1 = np.ones(len(data_X['Open']))
#     data_X = np.column_stack(
#         [test1, test2, data_X['High'].values.reshape(1, -1)[0], data_X['Low'].values.reshape(1, -1)[0], data_X['Volume'].values.reshape(1, -1)[0]])

#     print(test1.shape)
#     print(test2.shape)

#     data_Y = data['Close']
#     # data_Y = data_Y.values.reshape(1, -1)

#     print(data_X.shape)
#     # print(data_X)
#     print(data_Y.shape)
#     # print(data_Y)

#     train_X, test_X, train_y, test_y = train_test_split(
#         data_X, data_Y, test_size=0.5)

#     print('\n\nTraining Set')

#     # Creating the Regressor
#     regressor = LinearRegression()
#     regressor.fit(train_X, train_y)

#     # Make Predictions and Evaluate the results
#     predict_y = regressor.predict(test_X)
#     print('Prediction Score : ', regressor.score(test_X, test_y))
#     error = mean_squared_error(test_y, predict_y)
#     print('Mean Squared Error : ', error)

#     coeff = regressor.coef_
#     print("Coefficients: ", coeff)

#     # Plot the predicted and the expected values
#     # fig = plt.figure()
#     # ax = plt.axes()
#     # ax.grid()
#     # ax.set(xlabel='Close ($)', ylabel='Open ($)',
#     #        title='Panasonic Multiple Regression')
#     # ax.plot(test_X['Open'], test_y)
#     # ax.plot(test_X['Open'], predict_y)
#     # fig.savefig('LRPlot.png')
#     # plt.show()

#     return coeff, predict_y


# def simpleLinearRegression(df):
#     dates = []
#     for x in range(0, len(df["Date"])):
#         dates.append(x)
#     prices = df['Close']

#     dates = np.asanyarray(dates)
#     prices = np.asanyarray(prices)
#     dates = np.reshape(dates, (len(dates), 1))
#     prices = np.reshape(prices, (len(prices), 1))

#     xtrain, xtest, ytrain, ytest = train_test_split(
#         dates, prices, test_size=0.2)
#     # best = reg.score(ytrain, ytest)
#     best = 0
#     # bestReg
#     for _ in range(100):
#         xtrain, xtest, ytrain, ytest = train_test_split(
#             dates, prices, test_size=0.2)
#         reg = LinearRegression().fit(xtrain, ytrain)
#         acc = reg.score(xtest, ytest)
#         if acc > best:
#             best = acc
#             bestReg = reg

#     mean = 0
#     for i in range(10):
#         msk = np.random.rand(len(df)) < 0.8
#         xtest = dates[~msk]
#         ytest = prices[~msk]
#         mean += bestReg.score(xtest, ytest)

#     print("Average Accuracy: ", mean/10)

#     print("R^2 Score: ", bestReg.score(dates, prices))

#     # Plot Predicted VS Actual Data
#     # plt.plot(xtest, ytest, color='green', linewidth=1,
#     #          label='Actual Price')  # plotting the initial datapoints
#     # plt.plot(xtest, bestReg.predict(xtest), color='blue', linewidth=3,
#     #          label='Predicted Price')  # plotting the line made by linear regression
#     # plt.title('Linear Regression | Time vs. Price ')
#     # plt.legend()
#     # plt.xlabel('Date Integer')
#     # plt.show()

#     return bestReg.score(dates, prices)


# def polynomialRegression(df):
#     dates = []
#     for x in range(0, len(df["Date"])):
#         dates.append(x)

#     dates = np.array(dates)

#     prices = np.array(df['Close'])

#     x_train, x_test, y_train, y_test = train_test_split(
#         dates, prices, test_size=0.2)  # splits the dataset into 80-20 random points

#     print(x_train[0:5])
#     print(x_test[0:5])

#     x_train = np.array(x_train)
#     y_train = np.array(y_train)
#     x_test = np.array(x_test)
#     y_test = np.array(y_test)

#     x_train = x_train.reshape(-1, 1)
#     y_train = y_train.reshape(-1, 1)

#     y_train = y_train[x_train[:, 0].argsort()]
#     x_train = x_train[x_train[:, 0].argsort()]

#     # x_test = x_test.reshape(-1, 1)
#     y_test = y_test.reshape(-1, 1)

#     myList = np.array([])
#     for i in range(2, 21):
#         # Transform polynomial features
#         poly = PolynomialFeatures(degree=i, include_bias=False)
#         x_poly = poly.fit_transform(x_train)

#         # Train model
#         poly_reg_model = LinearRegression().fit(x_poly, y_train)

#         # Predict
#         y_predicted = poly_reg_model.predict(x_test_poly)

#         # Training error
#         score = poly_reg_model.score(x_poly, y_train)
#         # print("R^2 Score: ", score)
#         myList = np.append(myList, score)

#         # Plot data

#         # plt.title("Polynomial Regression", size=16)
#         plt.scatter(x_train, y_train, c='green')
#         plt.scatter(x_test, y_test, c='red')
#         # plt.plot(x_train, y_predicted, c="red")
#         # plt.show()

#     return myList


# Fetch data for Panasonic
# df_6752 = fetchCompanyData(6752, df)
# plotData(df_6752)

# r2score = simpleLinearRegression(df_6752)
# print(r2score)

# myList = polynomialRegression(df_6752)
# myList = np.insert(myList, 0, r2score)
# print(myList)
# # myList = myList * 100
# myList2 = []
# for x in range(1, 21):
#     myList2.append(x)


# plt.bar(myList2, myList, color='green')
# plt.xlabel('Degree')
# plt.ylabel('R^2 Score')
# plt.show()

# Plot data
# plotData(df_6752)
# plotData(df_6752)


# Selecting Multiple Security Code
# mask0 = df['SecuritiesCode'] == 1332
# mask1 = df['SecuritiesCode'] == 1333
# masks = mask0 | mask1
# targets = df.loc[masks]
# print(targets)


# --- NEW ---
# Fetch company data for Panasonic
df_6752 = fetchCompanyData(6752, df)

# SVR


# def SVR(df2):
#     df = df2[['Close']]
#     forecast_period = 30
#     df['Prediction'] = df[['Close']].shift(-forecast_period)
#     print(df.tail())

#     # Create numpy array of predictions
#     X = np.array(df.drop(['Prediction'], 1))

#     # Remove last 30 rows
#     X = X[:-forecast_period]
#     print(X[0:5])

#     y = np.array(df['Prediction'])
#     y = y[:-forecast_period]
#     print(y[0:5])

#     xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

#     # SVR
#     svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
#     svr_rbf.fit(xtrain, ytrain)

#     svm_confidence = svr_rbf.score(xtest, ytest)
#     print("SVM Confidence: ", svm_confidence)


def normalize_data(df):
    # df on input should contain only one column with the price data (plus dataframe index)
    min = df.min()
    max = df.max()
    x = df

    # time series normalization part
    # y will be a column in a dataframe
    y = (x - min) / (max - min)
    return y


def normalizeAndPlotData(df, list_of_companies):
    for i in range(0, len(list_of_companies) - 5):
        x = list_of_companies[i]
        df_single = fetchCompanyData(x['SecuritiesCode'], df)
        df_single.index = pd.to_datetime(df_single['Date'])
        df_single['norm'] = normalize_data(df_single['Close'])
        plt.plot(df_single['Date'], df_single['norm'])
        plt.legend(str(x['Name']), )

    plt.show()


# normalizeAndPlotData(df, list_of_companies)

def calculateSVR2(df):
    actual_price = df.tail()
    df = df.head(len(df) - 1)

    days = []
    close = []

    df_days = df.loc[:, 'Date']
    df_close = df.loc[:, 'Close']

    for i in range(len(df_days)):
        days.append(i)

    for close_price in df_close:
        close.append(float(close_price))

    days = np.array(days)
    days = days.reshape(-1, 1)
    # Create 3 SVR Models
    # Linear kernel
    lin_svr = SVR(kernel='linear', C=1000.0)
    lin_svr.fit(days, close)

    # Polynomial kernel
    poly_svr = SVR(kernel='poly', C=1000.0, degree=2)
    poly_svr.fit(days, close)

    # Radial kernel
    rbf_svr = SVR(kernel='rbf', C=10.0, gamma=0.15)
    rbf_svr.fit(days, close)

    R_score = rbf_svr.score(days, close)
    print(R_score)

    # Plot on graph
    plt.figure(figsize=(16, 8))
    plt.scatter(days, close, color='red', label='Data')
    plt.plot(days, rbf_svr.predict(days), color='green', label='RBF Model')
    plt.plot(days, poly_svr.predict(days),
             color='blue', label='Polynomial Model')
    plt.plot(days, lin_svr.predict(days), color='orange', label='Linear Model')
    plt.legend()
    plt.show()


def calculateRadialBasisSVR(df):
    actual_price = df.tail()
    df = df.head(len(df) - 1)

    days = []
    close = []

    df_days = df.loc[:, 'Date']
    df_close = df.loc[:, 'Close']

    for i in range(len(df_days)):
        days.append(i)

    for close_price in df_close:
        close.append(float(close_price))

    days = np.array(days)
    days = days.reshape(-1, 1)

    C_parameter = 10

    results = []

    for x in range(C_parameter, 200, 5):
        C_parameter = x
        # Radial kernel
        rbf_svr = SVR(kernel='rbf', C=float(C_parameter), gamma=0.15)
        rbf_svr.fit(days, close)
        R_score = rbf_svr.score(days, close)
        print("R^2 Score when C is " +
              str(C_parameter) + ": " + str(rbf_svr.score(days, close)))

        #   C -> C_parameter
        #   R_score -> R_score parameter
        results.append([C_parameter, R_score])

    return results


# radial_basis_accuracy = calculateRadialBasisSVR(df_6752)
# print(radial_basis_accuracy[0:5])
# x_val = []
# y_val = []
# for x in radial_basis_accuracy:
#     print(x)
#     x_val.append(x[0])
#     y_val.append(x[1])

# print(x_val[0:5])
# plt.plot(x_val, y_val)
# plt.xlabel('C Parameter')
# plt.ylabel('R^2 Score')
# plt.title('Relation between R^2 Score and C Parameter')
# plt.show()

# calculateSVR2(df_6752)

def generateRBFHeatmap(df):
    actual_price = df.tail()
    df = df.head(len(df) - 1)

    days = []
    close = []

    df_days = df.loc[:, 'Date']
    df_close = df.loc[:, 'Close']

    for i in range(len(df_days)):
        days.append(i)

    for close_price in df_close:
        close.append(float(close_price))

    days = np.array(days)
    days = days.reshape(-1, 1)

    # C_range = [10, 100, 200]
    C_range = np.arange(10, 105, 5)
    # 19

    # gamma_range = [0.10, 0.5, 1]
    gamma_range = np.arange(0.1, 1.05, 0.05)

    # param_grid = dict(gamma=gamma_range, C=C_range)
    # cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    # grid = GridSearchCV(SVR(), param_grid=param_grid, cv=cv)
    # grid.fit(days, close)

    table = np.array([[]])
    for C_parameter in C_range:
        row = np.array([])
        for gamma in gamma_range:
            rbf_svr = SVR(kernel='rbf', C=float(C_parameter), gamma=gamma)
            rbf_svr.fit(days, close)
            R_score = rbf_svr.score(days, close)
            print("C: ", C_parameter)
            print("Gamma: ", gamma)
            print("R_score: ", R_score)
            row = np.append(row, R_score)
        table = np.append(table, row)

    table = table.reshape(19, 19)
    print(table)
    plt.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(
        table,
        interpolation="nearest",
        cmap=plt.cm.hot,
        norm=MidpointNormalize(vmin=0.2, midpoint=0.65),
    )
    gamma_range = gamma_range.round(4)
    plt.xlabel("gamma")
    plt.ylabel("C")
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title("Validation accuracy")
    plt.show()


generateRBFHeatmap(df_6752)
