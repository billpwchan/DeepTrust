from anomaly_detection import *
from information_retrieval import *
from reliability_assessment import *
import argparse
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
    parser.add_argument('-adm', "--ad_method",
                        help="Specify Anomaly Detection method to use (ARIMA, Local Outlier Factor, Isolation Forest)",
                        type=str, choices=['arima', 'lof', 'if'])
    # Information Retrieval Arguments
    parser.add_argument('-ad', "--anomaly_date", help="Specify the anomaly date",
                        type=lambda d: datetime.strptime(d, '%d/%m/%Y').date())

    # Reliability Assessment Arguments
    parser.add_argument('-rat', "--ra_tasks", nargs='+', help="Specify Reliability Assessment tasks", type=str,
                        choices=['feature-filter', 'neural-generate', 'neural-update', 'neural-update-fake',
                                 'neural-verify'])
    parser.add_argument('-models', "--models", nargs='*', help="Specify Models for tasks", type=str,
                        choices=['gpt-2', 'gltr-gpt2', 'gltr-bert'])

    # Parse Arguments
    args = parser.parse_args()

    if args.module == 'AD' and \
            (args.ticker is None or args.start_date is None or args.end_date is None or args.ad_method is None):
        parser.error("Anomaly Detection requires --ticker, --start_date, --end_date, --ad_method")

    if args.module == 'IR' and (args.ticker is None or args.anomaly_date is None):
        parser.error("Information Retrieval requires --ticker, --anomaly_date")

    if args.module == 'RA' and (args.ticker is None or args.anomaly_date is None or args.ra_tasks is None):
        parser.error("Reliability Assessment requires --ticker, --anomaly_date --ra_tasks")

    if args.module == 'AD':
        ad_instance = AnomalyDetection(ticker=args.ticker, mode=args.ad_method)
        ad_instance.train()
        anomaly_list = ad_instance.detect(args.start_date, args.end_date)
        anomaly_summary = ad_instance.format_anomaly(anomaly_list)
        print(anomaly_summary)

    if args.module == 'IR':
        ir_instance = InformationRetrieval(input_date=args.anomaly_date, ticker=args.ticker)
        ir_instance.retrieve_tweets()

    if args.module == 'RA':
        ra_instance = ReliabilityAssessment(input_date=args.anomaly_date, ticker=args.ticker)
        if 'feature-filter' in args.ra_tasks:
            ra_instance.feature_filter()
        if 'neural-generate' in args.ra_tasks:
            ra_instance.neural_fake_news_dataset_handle()
            ra_instance.neural_fake_news_generator_fine_tune(model_type='gpt2', model_name_or_path='gpt2-medium')
            ra_instance.neural_fake_news_generation(model_type='gpt2',
                                                    model_name_or_path='./reliability_assessment/gpt_generator/')
        if 'neural-update' in args.ra_tasks:
            ra_instance.neural_fake_news_detection(gpt_2=('gpt-2' in args.models),
                                                   gltr_gpt2=('gltr-gpt2' in args.models),
                                                   gltr_bert=('gltr-bert' in args.models))
        if 'neural-update-fake' in args.ra_tasks:
            ra_instance.neural_fake_news_detection(gpt_2=('gpt-2' in args.models),
                                                   gltr_gpt2=('gltr-gpt2' in args.models),
                                                   gltr_bert=('gltr-bert' in args.models),
                                                   fake=True)


if __name__ == '__main__':
    main()
