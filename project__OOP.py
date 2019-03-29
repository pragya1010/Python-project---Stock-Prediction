import requests
import pandas as pd
import numpy as np
import pandas_datareader.data as pdb
from fbprophet import Prophet
import datetime
import csv
from itertools import zip_longest
import pygal
import matplotlib.pyplot as plt

class Stock_Data:
    def __init__(self, companysymbol, timeframe):
        self.symbol = companysymbol
        self.unit = timeframe[1]
        self.quantity = timeframe [0]

    def get_company(self):
        url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(self.symbol)
        result = requests.get(url).json()
        for x in result['ResultSet']['Result']:
            if x['symbol'] == self.symbol:
                return x['name']

    def get_stock_data(self):
        # print("Fetching stock prices: ", get_company(companyname))
        if self.unit == "months" or self.unit == "month":
            days = self.quantity * 30
        elif self.unit == "year" or self.unit == "years":
            days = self.quantity * 365
        else:
            days = 365

        # Starting date provided. We are taking 1 year data as. of now
        now = datetime.datetime.now()
        end = datetime.datetime(now.year, now.month, now.day)

        start = datetime.datetime.now() - datetime.timedelta(days=days)
        start = datetime.datetime(start.year, start.month, start.day)

        data = pdb.DataReader(self.symbol, 'yahoo', start, end)
        # filename = "/PROJECT/Data/"+self.symbol+ "_Stock_Data.csv"
        filename= "/Users/pragya/PycharmProjects/LAB/PROJECT/Data/"+self.symbol+ "_Stock_Data.csv"
        # print(filename)
        data.to_csv(filename)

        return data

class Stock_Predict:
    def __init__(self, df, num_days):
        self.stock_data = df
        self.num_days = int(num_days)

        if self.num_days == '':
            self.num_days = 10

    def prophet_model(self):

        stock_data = self.stock_data.filter(['Close'])

        # Prophet would need a feature column ds. Hence creating a new column with name ds which is the Date feature.
        stock_data['ds'] = stock_data.index

        # log transform the ‘Close’ variable to convert non-stationary data to stationary.
        stock_data['y'] = np.log(stock_data['Close'])

        # Using the Prophet model for analysis
        clf = Prophet()
        clf.fit(stock_data)

        self.ending_stock_price = round(stock_data['Close'][-1],2)

        future = clf.make_future_dataframe(periods=self.num_days)
        forecast = clf.predict(future)

        # print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

        forecast_plot = clf.plot(forecast)
        forecast_plot.show()

        # make the vizualization a little better to understand
        stock_data.set_index('ds', inplace=True)
        forecast.set_index('ds', inplace=True)

        stock_visual = stock_data.join(forecast[['yhat', 'yhat_lower', 'yhat_upper']], how='outer')

        # Visualize the original values .. non logarithmic.
        stock_visual['yhat_scaled'] = np.exp(stock_visual['yhat'])

        actual_data = stock_visual.Close.apply(lambda x: round(x, 2))
        forecasted_data = stock_visual.yhat_scaled.apply(lambda x: round(x, 2))
        date = future['ds']
        d = [date, actual_data, forecasted_data]

        readcsvdata = zip_longest(*d, fillvalue='')
        with open('/Users/pragya/PycharmProjects/LAB/PROJECT/Data/futurepredictions.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(("Date", "Actual_price", "Forecasted_Price"))
            wr.writerows(readcsvdata)
        myfile.close()

        # predicted days 10 days stock price
        self.prediction_data =  pd.read_csv('/Users/pragya/PycharmProjects/LAB/PROJECT/Data/futurepredictions.csv')
        self.prediction_data['Date'] = pd.to_datetime(self.prediction_data['Date']).apply(lambda x:x.strftime('%Y-%m-%d'))
        self.prediction_data = self.prediction_data[['Date', 'Forecasted_Price']]
        self.prediction_data.set_index('Date', inplace=True)
        self.future = self.prediction_data.tail(self.num_days)
        # next day predicted price
        self.next_price = self.prediction_data['Forecasted_Price'][len(self.prediction_data['Forecasted_Price'])- self.num_days]

        #### graph data
        graph = pygal.Line()
        graph.title = '%Prophet Model%'
        graph.x_labels = date
        graph.add('Actual data', actual_data)
        graph.add('Forecasted data', forecasted_data)
        graph_data = graph.render_data_uri()

        plt.plot(date, actual_data, color='red', label='Actual')
        plt.plot(date, forecasted_data, color='blue', label='Forecasted')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')

        plt.title('Model Predictions')
        plt.legend()
        plt.show()

        return graph_data


    def recommend(self):

        df = self.future
        suggestion = "Buy" if df['Forecasted_Price'][0]> self.ending_stock_price else "Sell"
        return suggestion

class Trend:
    def __init__(self, df):
        self.data = df

    def graph_analysis(self):
        self.data = self.data[['Open', 'High', 'Low', 'Close', 'Volume']]
        # define a new feature, HL_PCT
        self.data['HL_PCT'] = ((self.data['High'] - self.data['Low']) / self.data['Low']) * 100
        # define a new feature percentage change
        self.data['PCT_CHNG'] = ((self.data['Close'] - self.data['Open']) / self.data['Open']) * 100

        ## Price Trend = Graph1
        self.data['Close'].plot(figsize=(15, 6), color="green")
        plt.legend(loc=4)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title("Price Trend")
        plt.show()

        ## Moving Average = Graph2
        self.data['HL_PCT'].plot(figsize=(15, 6), color="red")
        plt.legend(loc=4)
        plt.xlabel('Date')
        plt.ylabel('High Low Percentage')
        plt.title("Moving Average")
        plt.show()

        ## Percentage Change= Graph1
        self.data['PCT_CHNG'].plot(figsize=(15, 6), color="blue")
        plt.legend(loc=4)
        plt.xlabel('Date')
        plt.ylabel('Percent Change')
        plt.title("Moving Percentage change")
        plt.show()

        graph = pygal.Line()
        graph.title = '%Price Trend%'
        graph.x_labels = self.data.index
        graph.add('Actual data', self.data['Close'])
        graph_data1 = graph.render_data_uri()

        ## Moving Average = Graph2
        graph = pygal.Line()
        graph.title = '%Moving Average%'
        graph.x_labels = self.data.index
        graph.add('Actual data', self.data['HL_PCT'])
        graph_data2 = graph.render_data_uri()

        ## Moving Percentage = Graph3
        graph = pygal.Line()
        graph.title = '%Moving Percentage%'
        graph.x_labels = self.data.index
        graph.add('Actual data', self.data['PCT_CHNG'])
        graph_data3 = graph.render_data_uri()

        return graph_data1, graph_data2, graph_data3


# ### GET STOCK DATA
# company = Stock_Data('AAPL', (1, "year"))
# print("Fetching {} {} data for {}".format(company.quantity,company.unit, company.get_company()))
# df = company.get_stock_data()
#
# #### TREND ANALYSIS
# trend = Trend(df)
# graph1, graph2, graph3 = trend.graph_analysis()
#
# ###### PREDICT
# predictor = Stock_Predict(df, 10)
# graph = predictor.prophet_model()
# df1 = predictor.future
# print(df1)
# print("Current Stock Price: ",predictor.ending_stock_price)
# print("Next Day Predicted Price:  ",predictor.next_price)
# print("Model recommends to ",predictor.recommend())
#
# def error():
#     # df = Stock_Data('AAPL', (1, "year"))
#     error_predictor = Stock_Predict(df[:len(df)-10], 10)
#     graph = error_predictor .prophet_model()
#     df2 = error_predictor.future
#     df1 = df.tail(10)
#     df_merged = df1.merge(df2, how='outer', left_index=True, right_index=True)
#     df_new = df_merged.dropna()[['Close', 'Forecasted_Price']]
#     mape = np.mean(np.abs((df_new['Close'] - df_new['Forecasted_Price']) / df_new['Close'])) * 100
#     print("Mean absolute percentage error : ", mape)
#
# error()