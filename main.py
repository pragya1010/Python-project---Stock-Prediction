from PROJECT.project__OOP import Stock_Data, Stock_Predict, Trend
import pandas as pd
import numpy as np

### GET STOCK DATA
company = Stock_Data('AAPL', (1, "year"))
print("Fetching {} {} data for {}".format(company.quantity,company.unit, company.get_company()))
df = company.get_stock_data()

#### TREND ANALYSIS
trend = Trend(df)
graph1, graph2, graph3 = trend.graph_analysis()

###### PREDICT
predictor = Stock_Predict(df, 10)
graph = predictor.prophet_model()
df1 = predictor.future
print(df1)
print("Current Stock Price: ",predictor.ending_stock_price)
print("Next Day Predicted Price:  ",predictor.next_price)
print("Model recommends to ",predictor.recommend())

def error():
    # df = Stock_Data('AAPL', (1, "year"))
    error_predictor = Stock_Predict(df[:len(df)-10], 10)
    graph = error_predictor .prophet_model()
    df2 = error_predictor.future
    df1 = df.tail(10)
    df_merged = df1.merge(df2, how='outer', left_index=True, right_index=True)
    df_new = df_merged.dropna()[['Close', 'Forecasted_Price']]
    mape = np.mean(np.abs((df_new['Close'] - df_new['Forecasted_Price']) / df_new['Close'])) * 100
    print("Mean absolute percentage error : ", mape)


