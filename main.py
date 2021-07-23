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
    parser.add_argument('-irt', "--ir_tasks", nargs='+', help="Specify Information Retrieval tasks", type=str,
                        choices=['tweet-search', 'tweet-update'])

    # Reliability Assessment Arguments
    parser.add_argument('-rat', "--ra_tasks", nargs='+', help="Specify Reliability Assessment tasks", type=str,
                        choices=['feature-filter',
                                 'neural-generate', 'neural-update',
                                 'neural-update-fake', 'neural-train', 'neural-verify',
                                 'subj-train', 'subj-update', 'subj-verify',
                                 'arg-update', 'arg-verify',
                                 'sentiment-verify', 'label', 'eval'])
    parser.add_argument('-models', "--models", nargs='*', help="Specify Models for tasks", type=str,
                        choices=['roberta', 'gltr-gpt2', 'gltr-bert', 'svm', 'infersent', 'textblob', 'wordemb',
                                 'ibm-fasttext'])

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
        if 'tweet-search' in args.ir_tasks:
            ir_instance.retrieve_tweets()
        if 'tweet-update' in args.ir_tasks:
            ir_instance.update_tweets()

    if args.module == 'RA':
        ra_instance = ReliabilityAssessment(input_date=args.anomaly_date, ticker=args.ticker)
        if 'feature-filter' in args.ra_tasks:
            ra_instance.feature_filter()
        if 'neural-generate' in args.ra_tasks:
            ra_instance.neural_fake_news_dataset_handle()
            ra_instance.neural_fake_news_generator_fine_tune(model_type='gpt2', model_name_or_path='gpt2-medium')
            ra_instance.neural_fake_news_generation(model_type='gpt2',
                                                    model_name_or_path='./reliability_assessment/neural_filter/gpt_generator/')
        if 'neural-train' in args.ra_tasks:
            if 'gltr-gpt2' in args.models:
                ra_instance.neural_fake_news_train_classifier(gltr_gpt2=('gltr-gpt2' in args.models))
            if 'gltr-bert' in args.models:
                ra_instance.neural_fake_news_train_classifier(gltr_bert=('gltr-bert' in args.models))
        if 'neural-update' in args.ra_tasks:
            if 'roberta' in args.models:
                ra_instance.neural_fake_news_update(roberta=('roberta' in args.models))
            if 'gltr-gpt2' in args.models:
                ra_instance.neural_fake_news_update(gltr_gpt2=('gltr-gpt2' in args.models))
            if 'gltr-bert' in args.models:
                ra_instance.neural_fake_news_update(gltr_bert=('gltr-bert' in args.models))
            if 'svm' in args.models:
                ra_instance.neural_fake_news_update(classifier=True)
        if 'neural-update-fake' in args.ra_tasks:
            if 'roberta' in args.models:
                ra_instance.neural_fake_news_update(roberta=('roberta' in args.models), fake=True)
            if 'gltr-gpt2' in args.models:
                ra_instance.neural_fake_news_update(gltr_gpt2=('gltr-gpt2' in args.models), fake=True)
            if 'gltr-bert' in args.models:
                ra_instance.neural_fake_news_update(gltr_bert=('gltr-bert' in args.models), fake=True)
        if 'neural-verify' in args.ra_tasks:
            ra_instance.neural_fake_news_verify()
        if 'subj-train' in args.ra_tasks:
            ra_instance.subjectivity_train(model_version=2)
        if 'subj-update' in args.ra_tasks:
            ra_instance.subjectivity_update(infersent=('infersent' in args.models),
                                            textblob=('textblob' in args.models),
                                            wordemb=('wordemb' in args.models), model_version=2)
        if 'subj-verify' in args.ra_tasks:
            ra_instance.subjectivity_verify()
        if 'arg-update' in args.ra_tasks:
            ra_instance.arg_update()
        if 'sentiment-verify' in args.ra_tasks:
            ra_instance.sentiment_verify()
        if 'label' in args.ra_tasks:
            ra_instance.tweet_label()
        if 'eval' in args.ra_tasks:
            ra_instance.tweet_eval()


if __name__ == '__main__':
    main()
