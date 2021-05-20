from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import plotly.graph_objs as go
import math
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from sklearn.metrics import mean_squared_error
from pmdarima.arima import auto_arima
import yfinance as yf
from pmdarima.arima import ADFTest
import pickle


class AnomalyDetection:
    def __init__(self, mode: str, ticker: str):
        self.mode = mode
        self.ticker = ticker
        self.time_series_df = pd.DataFrame()
        self.time_series_df = AnomalyDetection.__get_data(self.ticker)

    @staticmethod
    def __get_data(ticker: str):
        time_series_df = yf.download(tickers=ticker, period='max')
        time_series_df.reset_index(level=0, inplace=True)
        time_series_df = time_series_df[['Date', 'Close', 'Volume']]
        time_series_df.columns = time_series_df.columns.str.lower()
        return time_series_df

    def train(self):
        price_vals = self.time_series_df['close'].values
        price_log = np.log10(price_vals)

        train, test = price_vals[:-150], price_vals[-150:]
        train_log, test_log = np.log10(train), np.log10(test)

        adf_test = ADFTest(alpha=0.05)
        res, stationary = adf_test.should_diff(price_vals)

        stepwise_model = auto_arima(train_log, start_p=0, start_q=0,
                                    max_p=12, max_q=12, max_d=12,
                                    start_P=0, start_Q=0,
                                    max_P=12, max_D=12, max_Q=12,
                                    max_m=12, d=1, D=1, m=12,
                                    trace=True, stationary=stationary,
                                    error_action='ignore',
                                    suppress_warnings=True,
                                    stepwise=True, n_fits=50)

        with open(f'./anomaly_detection/{self.ticker}-arima.pkl', 'wb') as pkl:
            pickle.dump(stepwise_model, pkl)

    def detect(self, start_date: datetime, end_date: datetime):
        print("Hello")
