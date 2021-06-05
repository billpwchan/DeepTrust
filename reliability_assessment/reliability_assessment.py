import subprocess
import time
from datetime import date
import pathlib

import requests
import ast

from database.mongodb_atlas import MongoDB
from util import logger


class NeuralVerifier:
    def __init__(self):
        self.default_logger = logger.get_logger('neural_verifier')
        self.__download_models(mode='gpt-2')
        self.__init_gpt_model()

    def __init_gpt_model(self, model: str = 'detector-large.pt'):
        self.default_logger.info("Initialize GPT-2 Neural Verifier")
        gpt_2_server = subprocess.Popen(["python", "./reliability_assessment/gpt_detector/server.py",
                                         f"./reliability_assessment/gpt_detector/models/{model}"])
        time.sleep(10)
        self.default_logger.info("GPT-2 Neural Verifier Initialized")

    def __download_models(self, mode: str = 'gpt-2'):
        base_model = pathlib.Path("./reliability_assessment/gpt_detector/models/detector-base.pt")
        if not base_model.exists():
            open('./reliability_assessment/gpt_detector/models/detector-base.pt', 'wb').write(
                requests.get('https://openaipublic.azureedge.net/gpt-2/detector-models/v1/detector-base.pt').content)
            self.default_logger.info(f'{mode} base model downloaded')
        else:
            self.default_logger.info(f'{mode} base model exists')
        large_model = pathlib.Path("./reliability_assessment/gpt_detector/models/detector-large.pt")
        if not large_model.exists():
            open('./reliability_assessment/gpt_detector/models/detector-large.pt', 'wb').write(
                requests.get('https://openaipublic.azureedge.net/gpt-2/detector-models/v1/detector-large.pt').content)
            self.default_logger.info(f'{mode} large model downloaded')
        else:
            self.default_logger.info(f'{mode} large model exists')

    def detect(self, text, mode: str = 'gpt-2') -> dict:
        if mode == 'gpt-2':
            # Payload text should not have # symbols or it will ignore following text - less tokens
            url = f"http://localhost:8080/?={text.replace('#', '')}"
            payload = {}
            headers = {}
            response = requests.request("GET", url, headers=headers, data=payload)
            self.default_logger.info(response.text)
            # Return a dict representation of the returned text
            return ast.literal_eval(response.text)


class ReliabilityAssessment:
    def __init__(self, input_date: date, ticker: str):
        self.input_date = input_date
        self.ticker = ticker
        self.nv_instance = NeuralVerifier()
        self.db_instance = MongoDB()
        self.tweets_collection = self.db_instance.get_all_tweets(self.input_date, self.ticker)
        self.default_logger = logger.get_logger('reliability_assessment')

    def neural_fake_news_detection(self):
        for tweet in self.tweets_collection:
            self.nv_instance.detect(text=tweet['text'], mode='gpt-2')
