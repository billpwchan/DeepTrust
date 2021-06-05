import subprocess
import time
from datetime import date
import pathlib

import requests
import ast
import os
from database.mongodb_atlas import MongoDB
from util import logger


class NeuralVerifier:
    def __init__(self):
        self.default_logger = logger.get_logger('neural_verifier')
        self.__download_models(mode='gpt-2')
        self.__download_models(mode='grover')
        self.__init_gpt_model()

    def __init_gpt_model(self, model: str = 'detector-large.pt'):
        self.default_logger.info("Initialize GPT-2 Neural Verifier")
        gpt_2_server = subprocess.Popen(["python", "./reliability_assessment/gpt_detector/server.py",
                                         f"./reliability_assessment/gpt_detector/models/{model}"])
        time.sleep(10)
        self.default_logger.info("GPT-2 Neural Verifier Initialized")

    def __download_models(self, mode: str = 'gpt-2'):
        if mode == 'gpt-2':
            dir_prefix = "./reliability_assessment/gpt_detector/models/"
            base_model = pathlib.Path(f'{dir_prefix}detector-base.pt')
            if not base_model.exists():
                open(f'{dir_prefix}detector-base.pt', 'wb').write(
                    requests.get(
                        'https://openaipublic.azureedge.net/gpt-2/detector-models/v1/detector-base.pt').content)
                self.default_logger.info(f'{mode} base model downloaded')
            else:
                self.default_logger.info(f'{mode} base model exists')
            large_model = pathlib.Path(f'{dir_prefix}detector-large.pt')
            if not large_model.exists():
                open(f'{dir_prefix}detector-large.pt', 'wb').write(
                    requests.get(
                        'https://openaipublic.azureedge.net/gpt-2/detector-models/v1/detector-large.pt').content)
                self.default_logger.info(f'{mode} large model downloaded')
            else:
                self.default_logger.info(f'{mode} large model exists')
        if mode == 'grover':
            dir_prefix = "./reliability_assessment/grover/models/"
            model_type = 'discrimination'

            for ext in ['data-00000-of-00001', 'index', 'meta']:
                r = requests.get(
                    f'https://storage.googleapis.com/grover-models/{model_type}/generator=mega~discriminator=grover~discsize=mega~dataset=p=0.94/model.ckpt-1562.{ext}',
                    stream=True)
                with open(f'{dir_prefix}model.ckpt-1562.{ext}', 'wb') as f:
                    file_size = int(r.headers["content-length"])
                    if file_size < 1000:
                        raise ValueError("File doesn't exist? idk")
                    chunk_size = 1000
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                self.default_logger.info(f"{mode} {model_type}/model.ckpt.{ext} downloaded")

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
