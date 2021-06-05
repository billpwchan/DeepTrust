import subprocess
import time
from datetime import date

import requests

from database.mongodb_atlas import MongoDB
from util import logger


class NeuralVerifier:
    def __init__(self):
        self.default_logger = logger.get_logger('neural_verifier')
        self.__init_gpt_model()

    def __init_gpt_model(self, model: str = 'detector-large.pt'):
        self.default_logger.info("Initialize GPT-2 Neural Verifier")
        gpt_2_server = subprocess.Popen(["python", "./reliability_assessment/gpt_detector/server.py",
                                         f"./reliability_assessment/gpt_detector/{model}"])
        time.sleep(10)
        self.default_logger.info("GPT-2 Neural Verifier Initialized")

    def detect(self, text, mode: str = 'gpt-2'):
        if mode == 'gpt-2':
            # Payload text should not have # symbols or it will ignore following text - less tokens
            url = f"http://localhost:8080/?={text.replace('#', '')}"
            payload = {}
            headers = {}
            response = requests.request("GET", url, headers=headers, data=payload)
            self.default_logger.info(response.text)


class ReliabilityAssessment:
    def __init__(self, input_date: date, ticker: str):
        self.input_date = input_date
        self.ticker = ticker
        self.nv_instance = NeuralVerifier()
        self.db_instance = MongoDB()
        self.tweets_collection = self.db_instance.get_all_data(self.input_date, self.ticker)
        self.default_logger = logger.get_logger('reliability_assessment')

    def neural_fake_news_detection(self):
        for tweet in self.tweets_collection:
            self.nv_instance.detect(text=tweet['text'], mode='gpt-2')
