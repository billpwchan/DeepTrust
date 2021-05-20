import argparse

from anomaly_detection import *
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--module", help="Select which module to use (AD, IR, RA)", type=str,
                        choices=['AD', 'IR', 'RA'])
    # Anomaly Detection Arguments
    parser.add_argument("-t", "--ticker", help="Specify ticker", type=str)
    parser.add_argument('-sd', "--start_date", help="Specify the start date (DD/MM/YYYY)",
                        type=lambda d: datetime.strptime(d, '%d/%m/%Y').date())
    parser.add_argument('-ed', "--end_date", help="Specify the end date (DD/MM/YYYY)",
                        type=lambda d: datetime.strptime(d, '%d/%m/%Y').date())
    parser.add_argument('-adm', "--ad_method", help="Specify the end date", type=str, choices=['arima', 'lof', 'if'])
    args = parser.parse_args()

    if args.module == 'AD' and \
            (args.ticker is None or args.start_date is None or args.end_date is None or args.ad_method is None):
        parser.error("Anomaly Detection requires --ticker, --start_date, --end_date, --ad_method")

    if args.module == 'AD':
        ad_instance = AnomalyDetection(ticker=args.ticker, mode=args.ad_method)
        ad_instance.train()
        ad_instance.detect(args.start_date, args.end_date)


if __name__ == '__main__':
    main()
