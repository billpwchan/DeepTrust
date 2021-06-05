from datetime import date
from urllib import parse

from database.mongodb_atlas import MongoDB
from util import logger
import threading
import subprocess
import os
import requests


class NeuralVerifier:
    def __init__(self):
        self.default_logger = logger.get_logger('neural_verifier')
        self.default_logger.info("Initialize GPT-2 Neural Verifier")

    def __init_gpt_model(self):
        subprocess.run(["python", "./reliability_assessment/gpt_detector/server.py",
                        "./reliability_assessment/gpt_detector/detector-large.pt"], capture_output=True, check=True)
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
        self.db_instance = MongoDB()
        self.tweets_collection = self.db_instance.get_all_data(self.input_date, self.ticker)
        self.default_logger = logger.get_logger('reliability_assessment')
        self.nv_instance = NeuralVerifier()

    def neural_fake_news_detection(self):
        for tweet in self.tweets_collection:
            print(tweet['text'])
            self.nv_instance.detect(text=tweet['text'], mode='gpt-2')
