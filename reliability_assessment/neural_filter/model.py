import ast
import json
import pathlib
import subprocess
import time

import numpy as np
import requests

from util import logger

DETECTOR_MAP = {
    'detectors':            ['gpt-2'],
    'gpt-detector':         'detector-large.pt',
    'gltr-detector':        ('gpt2-xl', 'BERT'),
    'gpt-detector-server':  'http://localhost:8080/',
    'gltr-detector-server': ('http://localhost:5001/', 'http://localhost:5002/')
}
SUB_PROCESSES = []
PATH_RA = pathlib.Path.cwd() / 'reliability_assessment'
PATH_NEURAL = PATH_RA / 'neural_filter'


class NeuralVerifier:
    def __init__(self):
        self.default_logger = logger.get_logger('neural_verifier')
        for detector in DETECTOR_MAP['detectors']:
            self.__download_models(mode=detector)
        # python run_discrimination.py --input_data input_data.jsonl --output_dir models/mega-0.96 --config_file lm/configs/mega.json --predict_val true

    def init_gpt_model(self, model: str = DETECTOR_MAP['gpt-detector']):
        self.default_logger.info("Initialize GPT-2 Neural Verifier")
        gpt_2_server = subprocess.Popen(["python", str(PATH_NEURAL / 'roberta_detector' / 'server.py'),
                                         str(PATH_NEURAL / 'roberta_detector' / 'models' / model)])
        SUB_PROCESSES.append(gpt_2_server)
        while True:
            try:
                if requests.get(f"{DETECTOR_MAP['gpt-detector-server']}").status_code is not None:
                    self.default_logger.info("GPT-2 Neural Verifier Initialized")
                    break
            except requests.exceptions.ConnectionError:
                continue

    def init_gltr_models(self, model: str = DETECTOR_MAP['gltr-detector'][0]):
        if model not in DETECTOR_MAP['gltr-detector']:
            raise RuntimeError

        default_port = DETECTOR_MAP['gltr-detector-server'][DETECTOR_MAP['gltr-detector'].index(model)][-5:-1]
        self.default_logger.info(f"Initialize GLTR {model}")
        gltr_gpt_server = subprocess.Popen(
            ["python", str(PATH_NEURAL / 'gltr' / 'server.py'), "--model", model, "--port",
             f"{default_port}"])
        SUB_PROCESSES.append(gltr_gpt_server)
        while True:
            try:
                if requests.get(f'http://localhost:{default_port}/').status_code is not None:
                    self.default_logger.info(f"GLTR {model} Initialized")
                    break
            except requests.exceptions.ConnectionError:
                continue

    def __download_models(self, mode: str = 'gpt-2'):
        if mode == 'gpt-2':
            dir_prefix = PATH_NEURAL / 'roberta_detector' / 'models'
            base_model = pathlib.Path(dir_prefix / 'detector-base.pt')
            if not base_model.exists():
                open(str(base_model), 'wb').write(
                    requests.get(
                        'https://openaipublic.azureedge.net/gpt-2/detector-models/v1/detector-base.pt').content)
                self.default_logger.info(f'{mode} base model downloaded')
            else:
                self.default_logger.info(f'{mode} base model exists')
            large_model = pathlib.Path(dir_prefix / 'detector-large.pt')
            if not large_model.exists():
                open(str(large_model), 'wb').write(
                    requests.get(
                        'https://openaipublic.azureedge.net/gpt-2/detector-models/v1/detector-large.pt').content)
                self.default_logger.info(f'{mode} large model downloaded')
            else:
                self.default_logger.info(f'{mode} large model exists')

    def detect(self, text, mode: str = DETECTOR_MAP['gpt-detector']) -> dict or list:
        """
        Output Format for GPT-2: {'all_tokens', 'used_tokens', 'real_probability', 'fake_probability'}
        Output Format for GLTR: {'bpe_strings', 'pred_topk', 'real_topk', 'frac_hist'}
        Be noted that BERT may not return valid results due to empty tokenized text. Just ignore it.
        :param text: Tweet Text (Without Non-ASCII Code)
        :param mode: 'gpt-2' or 'gltr' currently supported
        :return:
        """
        if mode == DETECTOR_MAP['gpt-detector']:
            # Payload text should not have # symbols or it will ignore following text - less tokens
            url = f"{DETECTOR_MAP['gpt-detector-server']}?={text.replace('#', '')}"
            payload = {}
            headers = {}
            response = None
            for retry_limit in range(5):
                try:
                    response = requests.request("GET", url, headers=headers, data=payload)
                    break
                except requests.exceptions.ConnectionError:
                    time.sleep(1)
                    continue
            # self.default_logger.info(f'{mode}: {response.text}')
            # Return a dict representation of the returned text
            return ast.literal_eval(response.text) if response is not None else {}
        elif mode in DETECTOR_MAP['gltr-detector']:
            gltr_type = mode
            gltr_server = DETECTOR_MAP['gltr-detector-server'][DETECTOR_MAP['gltr-detector'].index(gltr_type)]
            url = f"{gltr_server}api/analyze"
            payload = json.dumps({
                "project": f"{gltr_type}",
                "text":    text
            })
            headers = {
                'Content-Type': 'application/json'
            }
            response = None
            for retry_limit in range(5):
                try:
                    response = requests.request("POST", url, headers=headers, data=payload)
                    break
                except requests.exceptions.ConnectionError:
                    time.sleep(1)
                    continue

            if response is not None and response.ok:
                gltr_result = json.loads(response.text)['result']
                # GLTR['result'].keys() = 'bpe_strings', 'pred_topk', 'real_topk'
                frac_distribution = [float(real_topk[1]) / float(gltr_result['pred_topk'][index][0][1])
                                     for index, real_topk in enumerate(gltr_result['real_topk'])]
                frac_histogram = np.histogram(frac_distribution, bins=10, range=(0.0, 1.0), density=False)
                gltr_result['frac_hist'] = frac_histogram[0].tolist()
                output_data = gltr_result
            else:
                self.default_logger.error(f'GLTR Exception: {payload}')
                output_data = {}
            return output_data
        else:
            raise NotImplementedError
