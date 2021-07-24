import atexit
import concurrent.futures
import configparser
import gc
import os
import re
import subprocess
import time
from datetime import date
from random import randint

import contractions
import emoji
import joblib
import nltk
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
import torch
from bert import BertModelLayer
from bert.tokenization.bert_tokenization import FullTokenizer
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from nltk.tokenize import TweetTokenizer
from profanity_check import predict_prob
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearnex import patch_sklearn
from textblob import TextBlob
from tqdm import tqdm, trange
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from database.mongodb_atlas import MongoDB
from reliability_assessment.neural_filter.gpt_generator.model import TweetGeneration
from reliability_assessment.neural_filter.model import DETECTOR_MAP, NeuralVerifier, SUB_PROCESSES
from reliability_assessment.sentiment_filter.finBERT.model import predict
from reliability_assessment.subj_filter.infersent.classifier import MLP
from reliability_assessment.subj_filter.infersent.model import InferSent
from reliability_assessment.subj_filter.wordemb.model import WordEmbPreprocess
from util import *

patch_sklearn()
gc.enable()
nltk.download('punkt')

PATH_RA = Path.cwd() / 'reliability_assessment'

PATH_NEURAL = PATH_RA / 'neural_filter'
PATH_SUBJ = PATH_RA / 'subj_filter'
PATH_SENTIMENT = PATH_RA / 'sentiment_filter'


class ReliabilityAssessment:
    def __init__(self, input_date: date, ticker: str):
        self.input_date = input_date
        self.ticker = ticker
        self.nv_instance = NeuralVerifier()
        self.db_instance = MongoDB()
        self.tg_instance = TweetGeneration()
        self.default_logger = logger.get_logger('reliability_assessment')
        self.config = configparser.ConfigParser()
        self.config.read('./config.ini')
        atexit.register(lambda: [p.kill() for p in SUB_PROCESSES])

    @staticmethod
    def __remove_non_ascii(text) -> str:
        return ''.join((c for c in text if 0 < ord(c) < 127))

    @staticmethod
    def __remove_twitter_link(text) -> str:
        return re.sub(r'https://t.co/[a-zA-Z0-9_.-]*$', '', text)

    def __tweet_feature_rules(self, tweet) -> bool:
        """
        TRUE means it satisfy the condition specified in the config. Should preserve records that are TRUE
        :param tweet:
        :return: bool
        """
        # For tweets that contain 20 or more cashtags, it is almost certain to be spam messages or stock updates.
        if len(re.findall(r'[$#][a-zA-Z]+', tweet['text'])) >= self.config.getint('RA.Feature.Config',
                                                                                  'max_tweet_tags'):
            return False

        # Need to have at least some interactions with the network
        public_metrics = tweet['public_metrics']
        if (public_metrics['retweet_count'] > self.config.getint('RA.Feature.Config', 'min_tweet_retweet') or
            public_metrics['reply_count'] > self.config.getint('RA.Feature.Config', 'min_tweet_reply') or
            public_metrics['like_count'] > self.config.getint('RA.Feature.Config', 'min_tweet_like') or
            public_metrics['quote_count'] > self.config.getint('RA.Feature.Config', 'min_tweet_quote')) and \
                predict_prob([tweet['text']])[0] < self.config.getfloat('RA.Feature.Config', 'max_profanity_prob') and \
                (('possibly_sensitive' not in tweet) or (not tweet['possibly_sensitive'])):
            return True
        return False

    def __author_feature_rules(self, author) -> bool:
        """
        TRUE means it satisfy the condition specified in the config. Should not remove those that return True.
        :param author:
        :return: bool
        """
        public_metrics = author['public_metrics']
        if public_metrics['followers_count'] > self.config.getint('RA.Feature.Config', 'min_author_followers') or \
                public_metrics['following_count'] > self.config.getint('RA.Feature.Config', 'min_author_following') or \
                public_metrics['tweet_count'] > self.config.getint('RA.Feature.Config', 'min_author_tweet') or \
                public_metrics['listed_count'] > self.config.getint('RA.Feature.Config', 'min_author_listed'):
            return True
        return False

    @staticmethod
    def __tweet_preprocess(text) -> str:
        """
        1. Remove leading and trailing spaces
        2. Remove useless Twitter link at the end
        3. Remove non-ascii characters that cannot be processed by detector.
        :param text:
        :return:
        """
        text = ReliabilityAssessment.__remove_twitter_link(text.strip())
        text = ReliabilityAssessment.__remove_non_ascii(text)
        return text

    def feature_filter(self):
        """
        Need to firstly filter out some information from the tweets collection.
        Remove tweets with no public_metrics, and authors with no public_metrics
        """
        # Always make a backup before doing any DB stuff!
        # self.db_instance.duplicate_collection(self.input_date, self.ticker, source='tweet', target='tweet_dump')

        # DON"T USE MONGO AGGREGATION. PYTHON IS MORE ROBUST
        projection_field = {'text': 1, 'author_id': 1, 'public_metrics': 1, 'possibly_sensitive': 1}
        tweets_collection = self.db_instance.get_all_tweets(self.input_date, self.ticker, database='tweet',
                                                            ra_raw=False, feature_filter=False,
                                                            projection_override=projection_field)
        self.default_logger.info(f"Tweet Collection: {len(tweets_collection)}")
        authors_collection = self.db_instance.get_all_authors(self.input_date, self.ticker, database='author')
        self.default_logger.info(f"Author Collection: {len(authors_collection)}")

        # Initialize Feature Records
        self.db_instance.update_all('ra_raw.feature-filter', False, self.input_date, self.ticker)

        # Append a field in the ra_raw.feature-filter
        authors_collection[:] = [author for author in authors_collection if self.__author_feature_rules(author)]
        authors_id = [author['id'] for author in authors_collection]
        tweets_collection[:] = [tweet for tweet in tqdm(tweets_collection) if
                                tweet['author_id'] in authors_id and self.__tweet_feature_rules(tweet)]

        batch_size = 60
        for i in trange(0, len(tweets_collection), batch_size):
            self.db_instance.update_one_bulk([tweet['_id'] for tweet in tweets_collection[i:i + batch_size]],
                                             'ra_raw.feature-filter',
                                             [True for _ in range(len(tweets_collection[i:i + batch_size]))],
                                             self.input_date, self.ticker)

    def detector_wrapper(self, tweet, mode):
        tweet_text = self.__tweet_preprocess(tweet['text'])
        return {'_id': tweet['_id'], 'output': self.nv_instance.detect(text=tweet_text, mode=mode)}

    def neural_fake_news_update(self, roberta: bool = False, gltr_gpt2: bool = False, gltr_bert: bool = False,
                                classifier: bool = False, fake: bool = False):
        # Always clean up fields before starting!
        # if input('CAUTION: DO YOU WANT TO CLEAN RA RESULTS? (Y/N) ') == "Y" and input('DOUBLE CHECK (Y/N) ') == 'Y':
        #     self.db_instance.remove_many('ra_raw', self.input_date, self.ticker)

        if roberta:
            self.nv_instance.init_gpt_model(model=DETECTOR_MAP['gpt-detector'])
            # Split large tweets collection into smaller pieces -> GOOD FOR LAPTOP :)
            batch_size = 30  # Good for 1080 Ti
            if fake:
                gpt_collection = \
                    self.db_instance.get_non_updated_tweets('ra_raw.RoBERTa-detector',
                                                            self.input_date, self.ticker,
                                                            database='fake',
                                                            select_field={"_id": 1, "id": 1, "text": 1},
                                                            feature_filter=False)
            else:
                gpt_collection = self.db_instance.get_non_updated_tweets('ra_raw.RoBERTa-detector',
                                                                         self.input_date, self.ticker)
            self.default_logger.info(f'Remaining entries to verify with GPT-2: {len(gpt_collection)}')

            for i in trange(0, len(gpt_collection), batch_size):
                tweets_collection_small = gpt_collection[i:i + batch_size]
                # Update RoBERTa-detector Results
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    gpt_2_futures = [executor.submit(self.detector_wrapper, tweet, DETECTOR_MAP['gpt-detector']) for
                                     tweet in tweets_collection_small]

                # Update MongoDB
                self.db_instance.update_one_bulk([future.result()['_id'] for future in gpt_2_futures],
                                                 'ra_raw.RoBERTa-detector',
                                                 [future.result()['output'] for future in gpt_2_futures],
                                                 self.input_date, self.ticker, database='fake' if fake else 'tweet')
                gc.collect()
            # Kill GPT-2 Process
            [p.kill() for p in SUB_PROCESSES]

        if gltr_gpt2 or gltr_bert:
            gltr_type = DETECTOR_MAP['gltr-detector'][0] if gltr_gpt2 else DETECTOR_MAP['gltr-detector'][1]
            self.nv_instance.init_gltr_models(model=gltr_type)
            batch_size = 2 if gltr_gpt2 else 50
            if fake:
                gltr_collection = self.db_instance.get_non_updated_tweets(
                    f"ra_raw.{gltr_type}-detector",
                    self.input_date, self.ticker,
                    database='fake',
                    select_field={"_id": 1, "id": 1, "text": 1},
                    feature_filter=False)
            else:
                gltr_collection = self.db_instance.get_non_updated_tweets(
                    f"ra_raw.{gltr_type}-detector", self.input_date, self.ticker)
            self.default_logger.info(f'Remaining entries to verify with GLTR: {len(gltr_collection)}')

            for i in trange(0, len(gltr_collection), batch_size):
                tweets_collection_small = gltr_collection[i:i + batch_size]
                # Update GLTR Results
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    gltr_futures = [executor.submit(self.detector_wrapper, tweet, gltr_type)
                                    for tweet in tweets_collection_small]

                self.db_instance.update_one_bulk([future.result()['_id'] for future in gltr_futures],
                                                 f"ra_raw.{gltr_type}-detector",
                                                 [future.result()['output'] for future in gltr_futures],
                                                 self.input_date, self.ticker, database='fake' if fake else 'tweet')

                gc.collect()
            [p.kill() for p in SUB_PROCESSES]

        if classifier:
            for gltr_type in DETECTOR_MAP['gltr-detector']:
                file_path = PATH_NEURAL / 'neural_classifier' / f'{self.ticker}_{self.input_date}_{gltr_type}_svm.pkl'
                file_path_check = Path(file_path)
                if file_path_check.is_file():
                    clf = joblib.load(file_path)
                else:
                    self.default_logger.error(f"Please train your SVM classifier first. Missing {file_path}")
                    return
                # Get all tweets with feature-filter = True
                tweets_collection = [tweet for tweet in
                                     self.db_instance.get_all_tweets(self.input_date, self.ticker, database='tweet',
                                                                     projection_override={
                                                                         f"ra_raw.{gltr_type}-detector.frac_hist": 1})
                                     if f'{gltr_type}-detector' in tweet['ra_raw'] and tweet['ra_raw'][
                                         f'{gltr_type}-detector']]

                # Classes Order: [0: Human, 1: Machine]
                batch_size = 100
                for i in trange(0, len(tweets_collection), batch_size):
                    tweets_collection_small = tweets_collection[i:i + batch_size]
                    y = clf.predict_proba(
                        [tweet['ra_raw'][f'{gltr_type}-detector']['frac_hist'] for tweet in tweets_collection_small])
                    self.db_instance.update_one_bulk([tweet['_id'] for tweet in tweets_collection_small],
                                                     f'ra_raw.{gltr_type}-detector.real_probability',
                                                     [prob[0] for prob in y], self.input_date, self.ticker)
                    self.db_instance.update_one_bulk([tweet['_id'] for tweet in tweets_collection_small],
                                                     f'ra_raw.{gltr_type}-detector.fake_probability',
                                                     [prob[1] for prob in y], self.input_date, self.ticker)

        self.default_logger.info("Neural Fake News Detector Output Update Success!")

    def neural_fake_news_dataset_handle(self):
        tweets_collection = [self.__tweet_preprocess(tweet['text']).replace("\n", "") for tweet in
                             self.db_instance.get_roberta_threshold_tweets(
                                 self.config.getfloat('RA.Neural.Config', 'roberta_threshold'),
                                 self.input_date, self.ticker)]

        train, test = np.split(np.array(tweets_collection), [int(len(tweets_collection) * 0.8)])
        for index, value in {'train': train, 'test': test}.items():
            with open(PATH_NEURAL / 'detector_dataset' / f'{self.ticker}_{self.input_date}_{index}.txt', 'w+',
                      encoding='utf-8') as file_handle:
                file_handle.writelines(f"{tweet}\n" for tweet in value)

    def neural_fake_news_generator_fine_tune(self, model_type, model_name_or_path):
        gpt_2_fine_tune = subprocess.call(
            ["python", str(PATH_NEURAL / 'run_clm.py'), "--model_type", "gpt2",
             "--model_name_or_path", "gpt2-medium",
             "--train_data_file",
             str(PATH_NEURAL / 'detector_dataset' / f'{self.ticker}_{self.input_date}_train.txt'),
             "--eval_data_file", str(PATH_NEURAL / 'detector_dataset' / f'{self.ticker}_{self.input_date}_test.txt'),
             "--line_by_line", "--do_train", "--do_eval", "--output_dir",
             str(PATH_NEURAL / 'gpt_generator' / f'{self.ticker}_{self.input_date}'), "--overwrite_output_dir",
             "--per_gpu_train_batch_size", "1", "--per_gpu_eval_batch_size", "1", "--learning_rate", "5e-5",
             "--save_steps", "50000", "--logging_steps", "50", "--num_train_epochs", "1"])

    def generator_wrapper(self, model_type, model_name_or_path, tweet) -> list:
        tweet_length = len(tweet['text'].split())
        return [{'text': individual_fake_tweet, 'original_id': tweet['id'], 'model': model_name_or_path}
                for individual_fake_tweet in
                self.tg_instance.tweet_generation(model_type=model_type,
                                                  model_name_or_path=model_name_or_path,
                                                  prompt=" ".join(tweet['text'].split()[
                                                                  :randint(2, max(2 + 1, int(tweet_length / 3)))]),
                                                  temperature=1, num_return_sequences=2, no_cuda=False)]

    def neural_fake_news_generation(self, model_type, model_name_or_path):
        """
        For each authentic tweet, generate a fake one based on a prompt (extracted from Top-..random substring in original tweet)
        :param model_type: ['gpt2', 'xlm']
        :param model_name_or_path: ['gpt2', 'gpt2-small', 'gpt2-medium', 'gpt2-xl', 'xlm-en-...']
        """
        model_name_or_path += f"{self.ticker}_{self.input_date}/"
        self.db_instance.drop_collection(self.input_date, self.ticker, database='fake')

        tweets_collection = [tweet for tweet in self.db_instance.get_roberta_threshold_tweets(
            self.config.getfloat('RA.Neural.Config', 'roberta_threshold'),
            self.input_date, self.ticker)]

        self.tg_instance.set_model(model_type, model_name_or_path)

        batch_size = 60
        for i in trange(0, len(tweets_collection), batch_size):
            tweets_collection_small = [tweet for tweet in tweets_collection[i:i + batch_size] if
                                       not self.db_instance.check_record_exists("original_id", tweet['id'],
                                                                                self.input_date,
                                                                                self.ticker, database='fake')]

            with concurrent.futures.ThreadPoolExecutor() as executor:
                tweet_futures = [executor.submit(self.generator_wrapper, model_type, model_name_or_path, tweet) for
                                 tweet in tweets_collection_small]

            self.db_instance.insert_many(self.input_date, self.ticker,
                                         [fake for future_result in tweet_futures for fake in
                                          future_result.result()], database='fake')

    @staticmethod
    def __frac_hist_handle(tweets_collection: list) -> list:
        output_list = []
        for entry in tweets_collection:
            output_list.append([value / sum(entry) for value in entry])
        return output_list

    def neural_fake_news_train_classifier(self, gltr_gpt2: bool = False, gltr_bert: bool = False,
                                          grid_search: bool = False):
        if gltr_gpt2 or gltr_bert:
            gltr_type = DETECTOR_MAP['gltr-detector'][0] if gltr_gpt2 else DETECTOR_MAP['gltr-detector'][1]
        else:
            return
        human_tweets_collection = [tweet['ra_raw'][f'{gltr_type}-detector']['frac_hist'] for tweet in
                                   self.db_instance.get_roberta_threshold_tweets(
                                       self.config.getfloat('RA.Neural.Config', 'roberta_threshold'),
                                       self.input_date, self.ticker, gltr={f"ra_raw.{gltr_type}-detector.frac_hist": 1})
                                   if f'{gltr_type}-detector' in tweet['ra_raw'] and
                                   tweet['ra_raw'][f'{gltr_type}-detector']]
        machine_tweets_collection = [tweet['ra_raw'][f'{gltr_type}-detector']['frac_hist'] for tweet in
                                     self.db_instance.get_all_tweets(self.input_date, self.ticker, database='fake',
                                                                     projection_override={
                                                                         f"ra_raw.{gltr_type}-detector.frac_hist": 1},
                                                                     feature_filter=False)
                                     if f'{gltr_type}-detector' in tweet['ra_raw'] and
                                     tweet['ra_raw'][f'{gltr_type}-detector']]
        self.default_logger.info(f'Human-Written Tweets Training Samples: {len(human_tweets_collection)}')
        self.default_logger.info(f'Machine-Written Tweets Training Samples: {len(machine_tweets_collection)}')

        human_tweets_collection = self.__frac_hist_handle(human_tweets_collection)
        machine_tweets_collection = self.__frac_hist_handle(machine_tweets_collection)

        X = human_tweets_collection + machine_tweets_collection
        y = ['Human' for _ in range(len(human_tweets_collection))] + \
            ['Machine' for _ in range(len(machine_tweets_collection))]

        df = pd.DataFrame(X)
        df['y'] = y
        df.to_csv(str(PATH_NEURAL / 'detector_dataset' / f'{self.ticker}_{self.input_date}_{gltr_type}.csv'))

        le = preprocessing.LabelEncoder()
        le.fit(y)
        self.default_logger.info(f'Label Encoder Classes: {le.classes_}')
        y = le.transform(y)

        if grid_search:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                                 'C':      [1, 10, 100, 1000]},
                                {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

            clf = GridSearchCV(SVC(), tuned_parameters, scoring='accuracy', n_jobs=-1, verbose=3, cv=10)
            clf.fit(X_train, y_train)
            self.default_logger.info(clf.best_params_)

            y_true, y_pred = y_test, clf.predict(X_test)
            self.default_logger.info(classification_report(y_true, y_pred))

        if gltr_gpt2:
            clf = SVC(C=1000, gamma=1, kernel='poly', degree=3)
        elif gltr_bert:
            clf = SVC(C=1000, gamma=1, kernel='rbf')
        else:
            # By far this configuration yields the best performance across two different test cases.
            clf = SVC(C=1000, gamma=1, kernel='rbf')

        calibrated_clf = CalibratedClassifierCV(base_estimator=clf, cv=3, n_jobs=-1)
        calibrated_clf.fit(X, y)
        joblib.dump(calibrated_clf,
                    str(PATH_NEURAL / 'neural_classifier' / f'{self.ticker}_{self.input_date}_{gltr_type}_svm.pkl'))

        # Evaluate the trained classifier for its performance
        self.default_logger.info(f'Calibrated Model Mean Accuracy: {calibrated_clf.score(X, y)}')

    def __neural_rules(self, roberta_prob: dict, gpt2_prob: dict, bert_prob: dict) -> bool or None:
        """
        The rule can be:
        1) If roberta_prob['real'] is greater than the threshold (e.g., 0.8), we say the tweet is definitely real
        2) If weighted average of gpt2 and bert['fake'] is greater than the threshold (e.g., 0.8), we say the tweet is definitely fake
        3) Otherwise, we say they are unknown / not conclusive
        :param roberta_prob: Real and Fake probability from the RoBERTa based Calibrated SVM Classifier
        :param gpt2_prob: Real and Fake probability from the GPT2-XL based Calibrated SVM Classifier
        :param bert_prob: Real and Fake probability from the BERT based Calibrated SVM Classifier
        :return:
        """
        # Empty dicts, mostly caused by tokenizer errors. Ignore them.
        if not roberta_prob or not gpt2_prob or not bert_prob:
            return False
        if roberta_prob['real_probability'] >= self.config.getfloat('RA.Neural.Config', 'roberta_threshold'):
            return True
        else:
            gpt2_weight = self.config.getfloat('RA.Neural.Config', 'gpt2_weight')
            bert_weight = self.config.getfloat('RA.Neural.Config', 'bert_weight')
            classifier_score = gpt2_weight * gpt2_prob['fake_probability'] + bert_weight * bert_prob['fake_probability']
            if classifier_score > self.config.getfloat('RA.Neural.Config', 'classifier_threshold') and \
                    roberta_prob['fake_probability'] >= self.config.getfloat('RA.Neural.Config', 'roberta_threshold'):
                return False

        neural_mode = self.config.get('RA.Neural.Config', 'neural_mode')
        assert neural_mode == '' or neural_mode == 'recall' or neural_mode == 'precision', "Invalid Neural Mode"
        return None if neural_mode == '' else True if neural_mode == 'recall' else False

    def neural_fake_news_verify(self):
        projection_field = {'ra_raw.BERT-detector.real_probability':    1,
                            'ra_raw.BERT-detector.fake_probability':    1,
                            'ra_raw.gpt2-xl-detector.real_probability': 1,
                            'ra_raw.gpt2-xl-detector.fake_probability': 1,
                            'ra_raw.RoBERTa-detector.real_probability': 1,
                            'ra_raw.RoBERTa-detector.fake_probability': 1}
        tweets_collection = self.db_instance.get_all_tweets(self.input_date, self.ticker, database='tweet',
                                                            ra_raw=False, feature_filter=True,
                                                            projection_override=projection_field)

        batch_size = 100
        for i in trange(0, len(tweets_collection), batch_size):
            tweets_collection_small = tweets_collection[i:i + batch_size]
            neural_filter = [self.__neural_rules(roberta_prob=tweet['ra_raw']['RoBERTa-detector'],
                                                 gpt2_prob=tweet['ra_raw']['gpt2-xl-detector'],
                                                 bert_prob=tweet['ra_raw']['BERT-detector'])
                             for tweet in tweets_collection_small]

            self.db_instance.update_one_bulk([tweet['_id'] for tweet in tweets_collection_small],
                                             'ra_raw.neural-filter', neural_filter, self.input_date, self.ticker)

    @staticmethod
    def __infersent_embeddings(model, batch, batch_size=8, tokenize=False) -> list:
        sentences = [' '.join(s) for s in batch]
        embeddings = model.encode(sentences, bsize=batch_size, tokenize=tokenize)
        return embeddings

    @staticmethod
    def __init_subjectivity_models(model_version: int = 2):
        MODEL_PATH = PATH_SUBJ / 'infersent' / 'encoder' / f'infersent{model_version}.pkl'
        W2V_PATH = PATH_SUBJ / 'infersent' / 'fastText' / 'crawl-300d-2M.vec' if model_version == 2 else PATH_SUBJ / 'infersent' / 'GloVe' / 'glove.840B.300d.txt'
        assert os.path.isfile(MODEL_PATH) and os.path.isfile(W2V_PATH), 'Please Set InferSent MODEL and W2V Paths'

        params_model = {'bsize':     64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                        'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
        infersent = InferSent(params_model)
        infersent.load_state_dict(torch.load(MODEL_PATH))
        infersent = infersent.cuda() if torch.cuda.is_available() else infersent
        infersent.set_w2v_path(W2V_PATH)
        return infersent

    def subjectivity_sentence_emb(self, model_version: int):
        infersent = self.__init_subjectivity_models(model_version)
        infersent.build_vocab_k_words(K=1999995)

        with open(PATH_SUBJ / 'infersent' / 'SUBJ' / 'subj.objective', 'r', encoding='latin-1') as f:
            obj = [line.split() for line in f.read().splitlines()]
        with open(PATH_SUBJ / 'infersent' / 'SUBJ' / 'subj.subjective', 'r', encoding='latin-1') as f:
            subj = [line.split() for line in f.read().splitlines()]

        # REMEMBER: OBJ = 1, SUB = 0
        samples, labels = obj + subj, [1] * len(obj) + [0] * len(subj)
        n_samples = len(samples)
        batch_size = 128
        enc_input = []
        # Sort to reduce padding
        sorted_corpus = sorted(zip(samples, labels), key=lambda z: (len(z[0]), z[1]))
        sorted_samples = [x for (x, y) in sorted_corpus]
        sorted_labels = [y for (x, y) in sorted_corpus]
        logging.info('Generating sentence embeddings')
        for ii in range(0, n_samples, batch_size):
            batch = sorted_samples[ii:ii + batch_size]
            embeddings = self.__infersent_embeddings(infersent, batch, batch_size)
            enc_input.append(embeddings)
        enc_input = np.vstack(enc_input)

        self.default_logger.info(f'Generated Sentence Embedding: {enc_input.shape}')
        return enc_input, sorted_labels

    def subjectivity_train(self, model_version):
        enc_input, sorted_labels = self.subjectivity_sentence_emb(model_version=2)

        config = {'nclasses':   2, 'seed': 1111,
                  'classifier': {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128, 'tenacity': 3, 'epoch_size': 2},
                  'nhid':       0, 'k_fold': 5}
        X = enc_input
        y = np.array(sorted_labels)

        regs = [10 ** t for t in range(-5, -1)]
        skf = StratifiedKFold(n_splits=config['k_fold'], shuffle=True, random_state=config['seed'])
        innerskf = StratifiedKFold(n_splits=config['k_fold'], shuffle=True, random_state=config['seed'])

        dev_results = []
        test_results = []
        count = 0
        opt_reg = 0
        for train_idx, test_idx in skf.split(X, y):
            count += 1
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            scores = []
            for reg in regs:
                regscores = []
                for inner_train_idx, inner_test_idx in innerskf.split(X_train, y_train):
                    X_in_train, X_in_test = X_train[inner_train_idx], X_train[inner_test_idx]
                    y_in_train, y_in_test = y_train[inner_train_idx], y_train[inner_test_idx]
                    clf = MLP(config['classifier'], inputdim=X.shape[1], nclasses=config['nclasses'], l2reg=reg,
                              seed=config['seed'])
                    clf.fit(X_in_train, y_in_train, validation_data=(X_in_test, y_in_test))
                    regscores.append(clf.score(X_in_test, y_in_test))
                scores.append(round(100 * np.mean(regscores), 2))
            opt_reg = regs[np.argmax(scores)]
            self.default_logger.info(
                f'Best param found at split {count}: l2reg = {opt_reg} with score {np.max(scores)}')
            dev_results.append(np.max(scores))

            clf = MLP(config['classifier'], inputdim=X.shape[1], nclasses=config['nclasses'], l2reg=opt_reg,
                      seed=config['seed'])
            clf.fit(X_train, y_train, validation_split=0.05)

            test_results.append(round(100 * clf.score(X_test, y_test), 2))

        dev_accuracy = round(np.mean(dev_results), 2)
        test_accuracy = round(np.mean(test_results), 2)
        self.default_logger.info(f'Dev Acc: {dev_accuracy}, Test Acc: {test_accuracy}')

        clf = MLP(config['classifier'], inputdim=X.shape[1], nclasses=config['nclasses'], l2reg=opt_reg,
                  seed=config['seed'])
        clf.fit(X, y, validation_split=0.05)
        joblib.dump(clf, PATH_SUBJ / 'infersent' / 'models' / f'{self.ticker}_{self.input_date}_mlp.pkl')

    @staticmethod
    def __enhanced_tweet_preprocess(text, text_processor) -> list:
        """
        Use default tweet preprocess technique first
        :param text:
        :return: a list of tokens using Tweet-specific tokenizer from NLTK
        """
        # Remove non-ascii characters + Remove irrelevant Twitter links
        text = ReliabilityAssessment.__tweet_preprocess(text)
        # Fix contractions (You're -> You are)
        text = contractions.fix(text)
        # Remove useless 's tags with no practical meanings
        text = text.replace("'s", '')
        # Convert Emoji to interpretable words (:smiley-faces)
        text = emoji.demojize(text, delimiters=("", ""))
        # Standard text preprocess defined in main function
        text = " ".join(text_processor.pre_process_doc(text))
        # Punctuations may carry subjective meanings for Infersent. Only remove functional punctuations
        text = re.sub("[^\w\s,.!?']", '', text.strip())
        # Remove excessive whitespaces
        text = re.sub(' +', ' ', text.strip())
        # NLTK Tweet Tokenizer to split texts
        text = TweetTokenizer().tokenize(text)
        return text

    @staticmethod
    def subj_wrapper(tweet) -> dict:
        textblob_obj = TextBlob(tweet['text']).sentiment
        return {'_id':    tweet['_id'],
                'output': {'subjectivity': textblob_obj.subjectivity, 'polarity': textblob_obj.polarity}}

    def subjectivity_update(self, infersent: bool = False, textblob: bool = False, wordemb: bool = False,
                            model_version: int = 2):
        """
        Verify text using sentence embedding.
        OBJ => 1, SUBJ => 0
        :param wordemb:
        :param textblob:
        :param infersent: supports ['infersent', 'textblob', 'bert-lstm']
        :param model_version: [1, 2] for infersent
        """
        text_processor = TextPreProcessor(
            # terms that will be normalized
            omit=['email', 'percent', 'money', 'phone', 'user', 'time', 'url', 'date', 'number'],
            annotate=[],
            fix_bad_unicode=True,  # fix HTML tokens
            segmenter="twitter",
            corrector="twitter",
            unpack_hashtags=True,  # perform word segmentation on hashtags
            unpack_contractions=True,  # Unpack contractions (can't -> can not)
            spell_correct_elong=False,  # spell correction for elongated words
            spell_correction=True,
            tokenizer=SocialTokenizer(lowercase=False).tokenize,
            dicts=[emoticons]
        )
        tweets_collection = self.db_instance.get_all_tweets(self.input_date, self.ticker, database='tweet',
                                                            ra_raw=False, feature_filter=True, neural_filter=False)

        if infersent:
            MODEL_PATH = PATH_SUBJ / 'infersent' / 'models' / f'{self.ticker}_{self.input_date}_mlp.pkl'
            assert os.path.isfile(MODEL_PATH), 'Please download the MLP model checkpoint'

            infersent = self.__init_subjectivity_models(model_version)
            infersent.build_vocab_k_words(K=1999995)

            clf = joblib.load(MODEL_PATH)

            batch_size = 128
            for i in trange(0, len(tweets_collection), batch_size):
                tweets_collection_small = tweets_collection[i:i + batch_size]
                tweets_text = [self.__enhanced_tweet_preprocess(tweet['text'], text_processor) for tweet in
                               tweets_collection_small]
                enc_input = np.vstack(self.__infersent_embeddings(infersent, tweets_text))

                result = [bool(output[0]) for output in clf.predict(enc_input)]
                self.db_instance.update_one_bulk([tweet['_id'] for tweet in tweets_collection_small],
                                                 'ra_raw.infersent-detector',
                                                 result, self.input_date, self.ticker)

        if textblob:
            batch_size = 128
            for i in trange(0, len(tweets_collection), batch_size):
                tweets_collection_small = tweets_collection[i:i + batch_size]
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    textblob_futures = [executor.submit(self.subj_wrapper, tweet) for tweet in tweets_collection_small]

                self.db_instance.update_one_bulk([future.result()['_id'] for future in textblob_futures],
                                                 f"ra_raw.textblob-detector",
                                                 [future.result()['output'] for future in textblob_futures],
                                                 self.input_date, self.ticker)
        if wordemb:
            checkpoint_path = PATH_SUBJ / 'wordemb' / 'models' / 'BERT_LSTM_CLR.h5'
            assert os.path.isfile(checkpoint_path), "Please download BERT_LSTM_CLR.h5 checkpoint from GitHub."
            bert_clr_lstm = tf.keras.models.load_model(checkpoint_path, compile=False,
                                                       custom_objects={'BertModelLayer': BertModelLayer,
                                                                       'Functional':     tf.keras.models.Model})
            bert_clr_lstm.compile(
                optimizer=tf.keras.optimizers.SGD(0.9),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]
            )
            tokenizer = FullTokenizer(vocab_file=str(PATH_SUBJ / 'wordemb' / 'models' / 'vocab.txt'))

            # REMEMBER: OBJ = 1, SUB = 0
            batch_size = 128
            for i in trange(0, len(tweets_collection), batch_size):
                tweets_collection_small = tweets_collection[i:i + batch_size]
                tweets_text = [" ".join(self.__enhanced_tweet_preprocess(tweet['text'], text_processor)) for tweet
                               in
                               tweets_collection_small]
                # Model Checkpoints is trained using max_seq_len of 128. With fine-tuned model, this data may be changed
                data = WordEmbPreprocess(X=tweets_text, y=None, tokenizer=tokenizer, max_seq_len=128)
                result = [bool(output) for output in list(np.argmax(bert_clr_lstm.predict(data.X, verbose=1), axis=1))]
                self.db_instance.update_one_bulk([tweet['_id'] for tweet in tweets_collection_small],
                                                 'ra_raw.wordemb-detector',
                                                 result, self.input_date, self.ticker)

    def __subjectivity_rules(self, infersent_output: bool, textblob_output: float, wordemb_output: bool) -> bool:
        if infersent_output and wordemb_output:
            return True
        elif infersent_output != wordemb_output:
            # If textblob_output < 0.5, then it is concluded as objective to break the tie between infersent and wordemb
            # If textblob == 0, ignore the results because it is likely the tokenizer from NLTK doesn't work
            return 0 < textblob_output < self.config.getfloat('RA.Subj.Config', 'textblob_threshold')
        # Both infersent and wordemb models output subjective, concluded as subjective.
        return False

    def subjectivity_verify(self):
        projection_field = {'ra_raw.infersent-detector':             1,
                            'ra_raw.textblob-detector.subjectivity': 1,
                            'ra_raw.wordemb-detector':               1}
        tweets_collection = self.db_instance.get_all_tweets(self.input_date, self.ticker, database='tweet',
                                                            ra_raw=False, feature_filter=True,
                                                            projection_override=projection_field)

        batch_size = 100
        for i in trange(0, len(tweets_collection), batch_size):
            tweets_collection_small = tweets_collection[i:i + batch_size]
            subj_filter = [self.__subjectivity_rules(
                infersent_output=tweet['ra_raw']['infersent-detector'],
                textblob_output=tweet['ra_raw']['textblob-detector']['subjectivity'],
                wordemb_output=tweet['ra_raw']['wordemb-detector'])
                for tweet in tweets_collection_small]

            self.db_instance.update_one_bulk([tweet['_id'] for tweet in tweets_collection_small],
                                             'ra_raw.subj-filter', subj_filter, self.input_date, self.ticker)

    def sentiment_verify(self, model='finBERT'):
        if model == 'finBERT':
            model_path = PATH_SENTIMENT / 'finBERT' / 'models' / 'finBERT_sentiment'
            model = AutoModelForSequenceClassification.from_pretrained(str(model_path), cache_dir=True)

            text_processor = TextPreProcessor(
                # terms that will be normalized
                omit=['email', 'percent', 'money', 'phone', 'user', 'time', 'url', 'date', 'number'],
                annotate=[],
                fix_bad_unicode=True,  # fix HTML tokens
                segmenter="twitter",
                corrector="twitter",
                unpack_hashtags=True,  # perform word segmentation on hashtags
                unpack_contractions=True,  # Unpack contractions (can't -> can not)
                spell_correct_elong=False,  # spell correction for elongated words
                spell_correction=True,
                tokenizer=SocialTokenizer(lowercase=True).tokenize,
                dicts=[emoticons]
            )

            tweets_collection = self.db_instance.get_all_tweets(self.input_date, self.ticker, database='tweet',
                                                                ra_raw=False, feature_filter=True, neural_filter=False)

            batch_size = 128
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            for i in trange(0, len(tweets_collection), batch_size):
                tweets_collection_small = tweets_collection[i:i + batch_size]
                tweets_text = [self.__enhanced_tweet_preprocess(tweet['text'], text_processor) for tweet in
                               tweets_collection_small]
                # Results should be a list of dataframe
                results = [predict(" ".join(tweet), model, tokenizer) for tweet in tweets_text]
                output = [{
                    'sentiment_score': result.iloc[0]['sentiment_score'].item(),
                    'prediction':      result.iloc[0]['prediction'],
                    'logit':           [np_float.item() for np_float in result.iloc[0]['logit']]
                } for result in results]

                self.db_instance.update_one_bulk([tweet['_id'] for tweet in tweets_collection_small],
                                                 'ra_raw.finBERT-detector',
                                                 output, self.input_date, self.ticker)

    @staticmethod
    def arg_wrapper(tweet) -> dict:
        models = ['IBMfasttext', 'PEdep', 'PEfasttext', 'PEglove', 'WDdep', 'WDfasttext', 'WDglove']
        payload = tweet['text']
        headers = {'Content-Type': 'text/plain'}
        output_dict = {'_id': tweet['_id'], 'output': {}}
        for model in models:
            url = f"http://ltdemos.informatik.uni-hamburg.de/arg-api//classify{model}"
            for retry_limit in range(5):
                try:
                    response = requests.request("POST", url, headers=headers, data=payload)
                    if response.status_code == 200:
                        output_dict['output'][model] = response.json()
                    break
                except:
                    time.sleep(5)
                    continue
        return output_dict

    def arg_update(self):
        text_processor = TextPreProcessor(
            # terms that will be normalized
            omit=['email', 'percent', 'money', 'phone', 'user', 'time', 'url', 'date', 'number'],
            annotate=[],
            fix_bad_unicode=True,  # fix HTML tokens
            segmenter="twitter",
            corrector="twitter",
            unpack_hashtags=True,  # perform word segmentation on hashtags
            unpack_contractions=True,  # Unpack contractions (can't -> can not)
            spell_correct_elong=False,  # spell correction for elongated words
            spell_correction=True,
            tokenizer=SocialTokenizer(lowercase=False).tokenize,
            dicts=[emoticons]
        )
        tweets_collection = self.db_instance.get_non_updated_tweets('ra_raw.targer-detector', self.input_date,
                                                                    self.ticker, database='tweet',
                                                                    select_field={'text': 1}, feature_filter=True)
        self.default_logger.info(f"Remaining Tweets: {len(tweets_collection)}")

        for tweet in tweets_collection:
            tweet['text'] = ReliabilityAssessment.__tweet_preprocess(tweet['text'])

        batch_size = 10
        for i in trange(0, len(tweets_collection), batch_size):
            tweets_collection_small = tweets_collection[i:i + batch_size]

            with concurrent.futures.ThreadPoolExecutor() as executor:
                targer_futures = [executor.submit(self.arg_wrapper, tweet) for tweet in tweets_collection_small]

            # Update MongoDB
            self.db_instance.update_one_bulk([future.result()['_id'] for future in targer_futures],
                                             'ra_raw.targer-detector',
                                             [future.result()['output'] for future in targer_futures],
                                             self.input_date, self.ticker)

    @staticmethod
    def __annotation_query(ticker) -> dict:
        stock_name = "twitter" if ticker == 'TWTR' else "facebook"
        # case-insensitive search.
        query_field = {"$or": [
            {"text": {"$regex": f".*\${ticker}.*", "$options": "i"}},
            {"$and": [
                {"text": {"$regex": f".*{stock_name}.*", "$options": "i"}},
                {"text": {"$regex": ".*stock.*", "$options": "i"}},
                {"text": {"$regex": ".*price.*", "$options": "i"}}
            ]}
        ]}
        return query_field

    def tweet_label(self):
        # tweets_collection = self.db_instance.get_all_tweets(self.input_date, self.ticker, database='tweet',
        #                                                     ra_raw=False, feature_filter=False, neural_filter=False)
        # output_tags = []
        # for tweet in tweets_collection:
        #     matches = re.findall(r'[$#][a-zA-Z]+', tweet['text'])
        #     if len(matches) > 10:
        #         output_tags.append(matches)
        #
        # with open(f"{self.ticker}_tags_pie.csv", "w", encoding="utf-8") as f:
        #     for record in output_tags:
        #         for item in record:
        #             f.write(f'{item}\n')
        #
        # exit(0)

        query_field = self.__annotation_query(self.ticker)
        label_dataset = self.db_instance.get_annotated_tweets(query_field, self.input_date, self.ticker)
        self.default_logger.info(f"Total Tweets: {len(label_dataset)}")
        label_dataset = [record for record in label_dataset if
                         'ra_raw' not in record or 'label' not in record['ra_raw']]
        self.default_logger.info(f"Remaining Tweets: {len(label_dataset)}")

        label_dataset = sorted(label_dataset, key=lambda k: k['text'])
        for tweet in label_dataset:
            self.db_instance.update_one(tweet['_id'], 'ra_raw.label', input(f'{tweet["text"]}:  ').lower() == "y",
                                        self.input_date, self.ticker)

    def tweet_eval(self):
        query_field = self.__annotation_query(self.ticker)

        filters = ['feature-filter', 'neural-filter', 'subj-filter', 'label']
        projection_filed = {f'ra_raw.{filter_name}': 1 for filter_name in filters}
        label_dataset = self.db_instance.get_annotated_tweets(query_field, self.input_date, self.ticker,
                                                              projection_override=projection_filed)

        eval_df = pd.DataFrame([item['ra_raw'] for item in label_dataset], columns=filters)

        eval_dict = {
            'feature':             eval_df['feature-filter'],
            'feature+neural':      eval_df['feature-filter'] & eval_df['neural-filter'],
            'feature+subj':        eval_df['feature-filter'] & eval_df['subj-filter'],
            'feature+neural+subj': eval_df['feature-filter'] & eval_df['neural-filter'] & eval_df['subj-filter']
        }
        for key, value in eval_dict.items():
            report = classification_report(eval_df['label'], value, output_dict=True)
            df = pd.DataFrame(report).transpose().to_csv(
                Path.cwd() / 'evaluation' / f'{self.ticker}_{self.input_date}_{key}.csv')
