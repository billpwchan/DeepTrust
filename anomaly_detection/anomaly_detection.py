import pickle
from datetime import date, datetime

import numpy as np
import pandas as pd
import yfinance as yf
from pmdarima.arima import ADFTest, auto_arima
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from tqdm import trange

pd.options.mode.chained_assignment = None


class AnomalyDetection:
    def __init__(self, ticker: str, mode: str = 'arima'):
        self.mode = mode
        self.ticker = ticker
        self.time_series_df = pd.DataFrame()
        self.time_series_df = AnomalyDetection.__get_data(self.ticker)
        self.price_vals = self.time_series_df['close'].values
        self.price_log = np.log10(self.price_vals)

    @staticmethod
    def __get_data(ticker: str):
        time_series_df = yf.download(tickers=ticker, period='max')
        time_series_df.reset_index(level=0, inplace=True)
        time_series_df = time_series_df[['Date', 'Close', 'Volume']]
        time_series_df.columns = time_series_df.columns.str.lower()
        time_series_df['date'] = time_series_df['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        return time_series_df

    def train(self):
        if self.mode == 'arima':
            adf_test = ADFTest(alpha=0.05)
            res, stationary = adf_test.should_diff(self.price_vals)

            stepwise_model = auto_arima(self.price_log, start_p=0, start_q=0,
                                        max_p=12, max_q=12, max_d=12,
                                        start_P=0, start_Q=0,
                                        max_P=12, max_D=12, max_Q=12,
                                        max_m=12, d=1, D=1, m=12, seasonal=False,
                                        trace=True, stationary=stationary, error_action='ignore',
                                        suppress_warnings=True,
                                        stepwise=True, n_fits=50)

            with open(f'./anomaly_detection/models/{self.ticker}-arima.pkl', 'wb') as pkl:
                pickle.dump(stepwise_model, pkl)
        elif self.mode == 'lof':
            lof = LocalOutlierFactor(n_neighbors=3)
            multivariate_df = self.time_series_df[['date', 'close', 'volume']]
            multivariate_df['date'] = multivariate_df['date'].apply(
                lambda x: datetime.timestamp(datetime.strptime(x, '%Y-%m-%d')))
            y_pred = lof.fit_predict(multivariate_df)
            # filter outlier index
            outlier_index = np.where(y_pred == -1)  # negative values are outliers and positives inliers
            # filter outlier values
            self.outlier_values = self.time_series_df.iloc[outlier_index]
        elif self.mode == 'if':
            clf = IsolationForest(max_features=3, n_estimators=500)
            multivariate_df = self.time_series_df[['date', 'close', 'volume']]
            multivariate_df['date'] = multivariate_df['date'].apply(
                lambda x: datetime.timestamp(datetime.strptime(x, '%Y-%m-%d')))
            y_pred = clf.fit_predict(multivariate_df)
            # filter outlier index
            outlier_index = np.where(y_pred == -1)  # negative values are outliers and positives inliers
            # filter outlier values
            self.outlier_values = self.time_series_df.iloc[outlier_index]
        else:
            raise NotImplementedError

    @staticmethod
    def __detect_classify_anomalies(output_df, window) -> pd.DataFrame:
        output_df.replace([np.inf, -np.inf], np.NaN, inplace=True)
        output_df.fillna(0, inplace=True)
        output_df['error'] = output_df['actual'] - output_df['predicted']
        output_df['percentage_change'] = ((output_df['actual'] - output_df['predicted']) / output_df['actual']) * 100
        output_df['meanval'] = output_df['error'].rolling(window=window).mean()
        output_df['deviation'] = output_df['error'].rolling(window=window).std()
        output_df['-3s'] = output_df['meanval'] - (2 * output_df['deviation'])
        output_df['3s'] = output_df['meanval'] + (2 * output_df['deviation'])
        output_df['-2s'] = output_df['meanval'] - (1.75 * output_df['deviation'])
        output_df['2s'] = output_df['meanval'] + (1.75 * output_df['deviation'])
        output_df['-1s'] = output_df['meanval'] - (1.5 * output_df['deviation'])
        output_df['1s'] = output_df['meanval'] + (1.5 * output_df['deviation'])
        cut_list = output_df[['error', '-3s', '-2s', '-1s', 'meanval', '1s', '2s', '3s']]
        cut_values = cut_list.values
        cut_sort = np.sort(cut_values)
        output_df['impact'] = [(lambda x: np.where(cut_sort == output_df['error'][x])[1][0])(x) for x in
                               range(len(output_df['error']))]
        severity = {0: 3, 1: 2, 2: 1, 3: 0, 4: 0, 5: 1, 6: 2, 7: 3}
        region = {0: "NEGATIVE", 1: "NEGATIVE", 2: "NEGATIVE", 3: "NEGATIVE", 4: "POSITIVE", 5: "POSITIVE",
                  6: "POSITIVE",
                  7: "POSITIVE"}
        output_df['color'] = output_df['impact'].map(severity)
        output_df['region'] = output_df['impact'].map(region)
        output_df['anomaly_points'] = np.where(output_df['color'] == 3, output_df['error'], np.nan)
        output_df = output_df.sort_values(by='date', ascending=False)
        output_df.date = pd.to_datetime(output_df['date'].astype(str), format="%Y-%m-%d")
        return output_df

    def detect(self, start_date: date, end_date: date) -> list:
        """
        Detect Anomalies within a given range and output a list of dates in %Y-%m-%d format (Datetime.date objects)
        :param start_date: Start Date in date() format
        :param end_date: End Date in date() format
        :return: A list of dates in %Y-%m-%d format
        """
        if self.mode == 'arima':
            stepwise_model = pickle.load(open(f'./anomaly_detection/models/{self.ticker}-arima.pkl', 'rb'))
            # HAVE TO SPECIFY A TRADING DAY?
            start_index = self.time_series_df.index[self.time_series_df['date'] == start_date.strftime('%Y-%m-%d')][0]
            end_index = self.time_series_df.index[self.time_series_df['date'] == end_date.strftime('%Y-%m-%d')][0]

            train, test = self.price_vals[:start_index], self.price_vals[start_index:end_index + 1]
            train_log, test_log = np.log10(train), np.log10(test)

            history = [x for x in train_log]
            predictions = list()
            predict_log = list()
            for t in trange(len(test_log)):
                stepwise_model.fit(history)
                output = stepwise_model.predict(n_periods=1)
                predict_log.append(output[0])
                predictions.append(10 ** output[0])
                history.append(test_log[t])

            predicted_df = pd.DataFrame()
            predicted_df['date'] = self.time_series_df['date'][start_index:end_index + 1]
            predicted_df['actual'] = test
            predicted_df['predicted'] = predictions
            predicted_df.reset_index(inplace=True, drop=True)

            classify_df = AnomalyDetection.__detect_classify_anomalies(predicted_df, 7)
            classify_df.reset_index(inplace=True, drop=True)
            classify_df.drop(classify_df.tail(7).index, inplace=True)
            output_data = classify_df[classify_df['anomaly_points'].notnull()]
            output_data.to_csv(f'./anomaly_detection/reports/{self.ticker}_arima_anomalies_{start_date}_{end_date}.csv')
        elif self.mode == 'lof' or self.mode == 'if':
            output_data = self.outlier_values[
                (self.outlier_values['date'] >= start_date.strftime('%Y-%m-%d')) &
                (self.outlier_values['date'] <= end_date.strftime('%Y-%m-%d'))]
        else:
            raise NotImplementedError

        output_data.to_csv(
            f'./anomaly_detection/reports/{self.ticker}_{self.mode}_anomalies_{start_date}_{end_date}.csv')
        return [timestamp.date() for timestamp in output_data['date'].tolist()]

    def format_anomaly(self, anomaly_list: list) -> dict:
        """
        Formats anomalies into a dictionary object. consists of ['ticker', 'name', 'date', 'quote_type'] attributes
        :param anomaly_list: List of Datetime.date objects
        :return: A dictionary of anomalies with relevant information
        """
        output_dict = {'date': anomaly_list, 'ticker': self.ticker}
        info = yf.Ticker(self.ticker).info
        output_dict['name'] = info['longName']
        output_dict['quote_type'] = info['quoteType'].lower()
        return output_dict
