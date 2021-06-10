import ast
import atexit
import concurrent.futures
import json
import pathlib
import subprocess
from datetime import date

import numpy as np
import requests
from tqdm import trange

from database.mongodb_atlas import MongoDB
from util import logger

SUB_PROCESSES = []

DETECTOR_MAP = {
    'detectors':            ('gpt-2', 'grover'),
    'gpt-detector':         'detector-large.pt',
    'gltr-detector':        ('gpt2-xl', 'BERT'),
    'gpt-detector-server':  'http://localhost:8080/',
    'gltr-detector-server': ('http://localhost:5001/', 'http://localhost:5002/')
}


class NeuralVerifier:
    def __init__(self):
        self.default_logger = logger.get_logger('neural_verifier')
        for detector in DETECTOR_MAP['detectors']:
            self.__download_models(mode=detector)
        # python run_discrimination.py --input_data input_data.jsonl --output_dir models/mega-0.96 --config_file lm/configs/mega.json --predict_val true

    def init_gpt_model(self, model: str = DETECTOR_MAP['gpt-detector']):
        self.default_logger.info("Initialize GPT-2 Neural Verifier")
        gpt_2_server = subprocess.Popen(["python", "./reliability_assessment/gpt_detector/server.py",
                                         f"./reliability_assessment/gpt_detector/models/{model}"])
        SUB_PROCESSES.append(gpt_2_server)
        while True:
            try:
                if requests.get(f"{DETECTOR_MAP['gpt-detector-server']}").status_code is not None:
                    self.default_logger.info("GPT-2 Neural Verifier Initialized")
                    break
            except requests.exceptions.ConnectionError:
                continue

    def init_gltr_models(self, models: tuple = DETECTOR_MAP['gltr-detector']):
        default_port = 5001
        for model in models:
            self.default_logger.info(f"Initialize GLTR {model}")
            gltr_gpt_server = subprocess.Popen(
                ["python", "./reliability_assessment/gltr/server.py", "--model", f"{model}", "--port",
                 f"{default_port}"])
            SUB_PROCESSES.append(gltr_gpt_server)
            while True:
                try:
                    if requests.get(f'http://localhost:{default_port}/').status_code is not None:
                        self.default_logger.info(f"GLTR {model} Initialized")
                        default_port += 1
                        break
                except requests.exceptions.ConnectionError:
                    continue

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
            dir_prefix = "./reliability_assessment/grover/models/mega-0.94/"
            model_type = 'discrimination'
            for ext in ['data-00000-of-00001', 'index', 'meta']:
                model_path = pathlib.Path(f'{dir_prefix}model.ckpt-1562.{ext}')
                if not model_path.exists():
                    r = requests.get(
                        f'https://storage.googleapis.com/grover-models/{model_type}/generator=mega~discriminator=grover~discsize=mega~dataset=p=0.94/model.ckpt-1562.{ext}',
                        stream=True)
                    with open(f'{dir_prefix}model.ckpt-1562.{ext}', 'wb') as f:
                        file_size = int(r.headers["content-length"])
                        if file_size < 1000:
                            raise ValueError("File doesn't exist")
                        chunk_size = 1000
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            f.write(chunk)
                    self.default_logger.info(f"{mode} {model_type}/model.ckpt.{ext} downloaded")
                else:
                    self.default_logger.info(f"{mode} {model_type}/model.ckpt.{ext} exists")

    def detect(self, text, mode: str = 'gpt-2') -> dict or list:
        """
        Output Format for GPT-2: {'all_tokens', 'used_tokens', 'real_probability', 'fake_probability'}
        Output Format for GLTR: {'bpe_strings', 'pred_topk', 'real_topk', 'frac_hist'}
        Be noted that BERT may not return valid results due to empty tokenized text. Just ignore it.
        :param text: Tweet Text (Without Non-ASCII Code)
        :param mode: 'gpt-2' or 'gltr' currently supported
        :return:
        """
        if mode == 'gpt-2':
            # Payload text should not have # symbols or it will ignore following text - less tokens
            url = f"{DETECTOR_MAP['gpt-detector-server']}?={text.replace('#', '')}"
            payload = {}
            headers = {}
            response = requests.request("GET", url, headers=headers, data=payload)
            # self.default_logger.info(f'{mode}: {response.text}')
            # Return a dict representation of the returned text
            return ast.literal_eval(response.text)
        elif mode == 'gltr':
            output_data = []
            for gltr_type, gltr_server in zip(DETECTOR_MAP['gltr-detector'], DETECTOR_MAP['gltr-detector-server']):
                url = f"{gltr_server}api/analyze"
                payload = json.dumps({
                    "project": f"{gltr_type}",
                    "text":    text
                })
                headers = {
                    'Content-Type': 'application/json'
                }
                response = requests.request("POST", url, headers=headers, data=payload)
                if response.ok:
                    gltr_result = json.loads(response.text)['result']
                    # GLTR['result'].keys() = 'bpe_strings', 'pred_topk', 'real_topk'
                    frac_distribution = [float(real_topk[1]) / float(gltr_result['pred_topk'][index][0][1])
                                         for index, real_topk in enumerate(gltr_result['real_topk'])]
                    frac_histogram = np.histogram(frac_distribution, bins=10, range=(0.0, 1.0), density=False)
                    gltr_result['frac_hist'] = frac_histogram[0].tolist()
                    # self.default_logger.info(f'{gltr_type}: {frac_perc_distribution}')
                    output_data.append(gltr_result)
                else:
                    self.default_logger.error(f'GLTR Exception: {payload}')
                    output_data.append({})
            return output_data
        else:
            raise NotImplementedError


class ReliabilityAssessment:
    def __init__(self, input_date: date, ticker: str):
        self.input_date = input_date
        self.ticker = ticker
        self.nv_instance = NeuralVerifier()
        self.db_instance = MongoDB()
        self.tweets_collection = self.db_instance.get_all_tweets(self.input_date, self.ticker)
        self.default_logger = logger.get_logger('reliability_assessment')

        atexit.register(lambda: [p.kill() for p in SUB_PROCESSES])

    @staticmethod
    def __remove_non_ascii(text):
        return ''.join((c for c in text if 0 < ord(c) < 127))

    def detector_wrapper(self, tweet, mode):
        tweet_text = self.__remove_non_ascii(tweet['text'])
        return {'_id': tweet['_id'], 'output': self.nv_instance.detect(text=tweet_text, mode=mode)}

    def neural_fake_news_detection(self, gpt_2: bool, gltr: bool):
        # Always clean up fields before starting!
        if input('CAUTION: DO YOU WANT TO CLEAN RA RESULTS? (Y/N) ') == "Y" and input('DOUBLE CHECK (Y/N) ') == 'Y':
            self.db_instance.remove_many('ra_raw', self.input_date, self.ticker)

        if gpt_2:
            self.nv_instance.init_gpt_model(model=DETECTOR_MAP['gpt-detector'])
            # Split large tweets collection into smaller pieces -> GOOD FOR LAPTOP :)
            SLICES = 10
            gpt_collection = [tweet for tweet in self.tweets_collection if
                              not ('ra_raw' in tweet and 'RoBERTa-detector' in tweet['ra_raw'])]
            self.default_logger.info(f'Remaining entries to verify with GPT-2: {len(gpt_collection)}')

            for i in trange(0, len(gpt_collection), SLICES):
                tweets_collection_small = gpt_collection[i:i + SLICES]
                # Update RoBERTa-detector Results
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    gpt_2_futures = [executor.submit(self.detector_wrapper, tweet, 'gpt-2') for tweet in
                                     tweets_collection_small]

                # Update MongoDB
                for future in gpt_2_futures:
                    self.db_instance.update_one(future.result()['_id'], 'ra_raw.RoBERTa-detector',
                                                future.result()['output'],
                                                self.input_date, self.ticker)
            # Kill GPT-2 Process
            [p.kill() for p in SUB_PROCESSES]

        if gltr:
            self.nv_instance.init_gltr_models(models=DETECTOR_MAP['gltr-detector'])
            SLICES = 3
            gltr_collection = [tweet for tweet in self.tweets_collection if
                               not ('ra_raw' in tweet and
                                    f"{DETECTOR_MAP['gltr-detector'][0]}-detector" in tweet['ra_raw'] and
                                    f"{DETECTOR_MAP['gltr-detector'][1]}-detector" in tweet['ra_raw'])
                               ]
            self.default_logger.info(f'Remaining entries to verify with GLTR: {len(gltr_collection)}')

            for i in trange(0, len(gltr_collection), SLICES):
                tweets_collection_small = gltr_collection[i:i + SLICES]
                # Update GLTR Results
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    gltr_futures = [executor.submit(self.detector_wrapper, tweet, 'gltr') for tweet in
                                    tweets_collection_small]
                for future in gltr_futures:
                    self.db_instance.update_one(future.result()['_id'],
                                                f"ra_raw.{DETECTOR_MAP['gltr-detector'][0]}-detector",
                                                future.result()['output'][0], self.input_date, self.ticker)
                    self.db_instance.update_one(future.result()['_id'],
                                                f"ra_raw.{DETECTOR_MAP['gltr-detector'][1]}-detector",
                                                future.result()['output'][1], self.input_date, self.ticker)
            [p.kill() for p in SUB_PROCESSES]

        self.default_logger.info("Neural Fake News Detector Output Update Success!")
