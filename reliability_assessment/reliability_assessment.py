from datetime import date

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
        # subprocess.run(["python", "./reliability_assessment/gpt_detector/server.py",
        #                 "./reliability_assessment/gpt_detector/detector-large.pt"], capture_output=True, check=True)
        # subprocess.check_output(["python", "./reliability_assessment/gpt_detector/server.py",
        #                          "./reliability_assessment/gpt_detector/detector-large.pt"])
        os.system(
            'python ./reliability_assessment/gpt_detector/server.py ./reliability_assessment/gpt_detector/detector-large.pt')

    def __init_gpt_model(self):
        self.gpt_server_thread = threading.Thread()
        self.gpt_server_thread.start()

    def __stop_gpt_model(self):
        self.gpt_server_thread.raise_exception()

    def detect_text(self, text, mode: str = 'gpt-2'):
        if mode == 'gpt-2':
            url = f"localhost:8080?={text}"
            payload = {}
            headers = {}
            response = requests.request("GET", url, headers=headers, data=payload)
            print(response.text)


class ReliabilityAssessment:
    def __init__(self, input_date: date, ticker: str):
        self.input_date = input_date
        self.ticker = ticker
        self.db_instance = MongoDB()
        # self.tweets_collection = self.db_instance.get_all_data(self.input_date, self.ticker)
        self.default_logger = logger.get_logger('reliability_assessment')
        self.nv_instance = NeuralVerifier()

    def neural_fake_news_detection(self):
        print("HELLO")
