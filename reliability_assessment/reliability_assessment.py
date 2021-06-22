import ast
import atexit
import concurrent.futures
import configparser
import gc
import json
import pathlib
import re
import subprocess
import time
from datetime import date
from random import randint

import joblib
import numpy as np
import pandas as pd
import requests
import torch
from profanity_check import predict_prob
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from tqdm import trange
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
)

from database.mongodb_atlas import MongoDB
from util import *

gc.enable()

SUB_PROCESSES = []

DETECTOR_MAP = {
    'detectors':            'gpt-2',
    'gpt-detector':         'detector-large.pt',
    'gltr-detector':        ('gpt2-xl', 'BERT'),
    'gpt-detector-server':  'http://localhost:8080/',
    'gltr-detector-server': ('http://localhost:5001/', 'http://localhost:5002/')
}


class TweetGeneration:
    default_logger = logger.get_logger('tweet_generation')

    MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

    MODEL_CLASSES = {
        "gpt2":       (GPT2LMHeadModel, GPT2Tokenizer),
        "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
        "xlm":        (XLMWithLMHeadModel, XLMTokenizer),
    }

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.args = None

    def set_model(self, model_type, model_name_or_path, no_cuda=False, fp16=False):
        if model_type not in TweetGeneration.MODEL_CLASSES.keys():
            raise RuntimeError(f'NEED TO BE ONE OF {TweetGeneration.MODEL_CLASSES.keys()}')

        args = {
            'model_type':         model_type,
            'model_name_or_path': model_name_or_path,
            'no_cuda':            no_cuda,
            'fp16':               fp16
        }
        # Pass dot reference check!
        args = dotdict(args)

        args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

        TweetGeneration.default_logger.warning(
            f"device: {args.device}, n_gpu: {args.n_gpu}, 16-bits training: {args.fp16}")

        # TweetGeneration.set_seed(args)
        # Initialize the model and tokenizer
        try:
            args.model_type = args.model_type.lower()
            model_class, tokenizer_class = TweetGeneration.MODEL_CLASSES[args.model_type]
        except KeyError:
            raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

        self.tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
        self.model = model_class.from_pretrained(args.model_name_or_path)
        self.model.to(args.device)

        self.args = args

    @staticmethod
    def set_seed(args):
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)

    @staticmethod
    def prepare_xlm_input(args, model, tokenizer, prompt_text):
        # kwargs = {"language": None, "mask_token_id": None}

        # Set the language
        use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
        if hasattr(model.config, "lang2id") and use_lang_emb:
            available_languages = model.config.lang2id.keys()
            if args.xlm_language in available_languages:
                language = args.xlm_language
            else:
                language = None
                while language not in available_languages:
                    language = input("Using XLM. Select language in " + str(list(available_languages)) + " >>> ")

            model.config.lang_id = model.config.lang2id[language]
            # kwargs["language"] = tokenizer.lang2id[language]

        # XLM masked-language modeling (MLM) models need masked token
        # is_xlm_mlm = "mlm" in args.model_name_or_path
        # if is_xlm_mlm:
        #     kwargs["mask_token_id"] = tokenizer.mask_token_id

        return prompt_text

    PREPROCESSING_FUNCTIONS = {
        "xlm": prepare_xlm_input,
    }

    @staticmethod
    def adjust_length_to_model(length, max_sequence_length):
        if length < 0 < max_sequence_length:
            length = max_sequence_length
        elif 0 < max_sequence_length < length:
            length = max_sequence_length  # No generation bigger than model size
        elif length < 0:
            length = TweetGeneration.MAX_LENGTH  # avoid infinite loop
        return length

    def tweet_generation(self, model_type, model_name_or_path, prompt="", length=50, stop_token=None, temperature=1.0,
                         repetition_penalty=1.0, k=0, p=0.9, prefix="", xlm_language="", seed=42, no_cuda=False,
                         num_return_sequences=10, fp16=False) -> list:
        """

        :param model_type: Model type ('gpt2', 'openai-gpt', 'xlm')
        :param model_name_or_path: Path to pre-trained model or shortcut name ('gpt2', 'openai-gpt', 'xlm')
        :param prompt:
        :param length: Length of generated sequence (Tweet should be around 20)
        :param stop_token: Token at which text generation is stopped
        :param temperature: Temperature of 1.0 has no effect, lower tend toward greedy sampling
                            Temperature -> Boltzmann distribution - Sampling deterministic
        :param repetition_penalty: Not useful for gpt-2 model
        :param k: Sampling range
        :param p:
        :param prefix:
        :param xlm_language:
        :param seed:
        :param no_cuda:
        :param num_return_sequences: The number of samples to generate.
        :param fp16:
        :return: a list of generated tweets!
        """
        if model_type not in TweetGeneration.MODEL_CLASSES.keys():
            raise RuntimeError(f'NEED TO BE ONE OF {TweetGeneration.MODEL_CLASSES.keys()}')

        args = {
            'model_type':           model_type,
            'model_name_or_path':   model_name_or_path,
            'prompt':               prompt,
            'length':               length,
            'stop_token':           stop_token,
            'temperature':          temperature,
            'repetition_penalty':   repetition_penalty,
            'k':                    k,
            'p':                    p,
            'prefix':               prefix,
            'xlm_language':         xlm_language,
            'seed':                 seed,
            'no_cuda':              no_cuda,
            'num_return_sequences': num_return_sequences,
            'fp16':                 fp16
        }

        for index, value in self.args.items():
            args[index] = value

        # Pass dot reference check!
        args = dotdict(args)

        if args.fp16:
            self.model.half()

        args.length = TweetGeneration.adjust_length_to_model(args.length,
                                                             max_sequence_length=self.model.config.max_position_embeddings)

        prompt_text = args.prompt if args.prompt else " "

        # Different models need different input formatting and/or extra arguments
        requires_preprocessing = args.model_type in TweetGeneration.PREPROCESSING_FUNCTIONS.keys()
        if requires_preprocessing:
            prepare_input = TweetGeneration.PREPROCESSING_FUNCTIONS.get(args.model_type)
            preprocessed_prompt_text = TweetGeneration.prepare_xlm_input(args, self.model, self.tokenizer, prompt_text)

            if self.model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
                tokenizer_kwargs = {"add_space_before_punct_symbol": True}
            else:
                tokenizer_kwargs = {}

            encoded_prompt = self.tokenizer.encode(
                preprocessed_prompt_text, add_special_tokens=False, return_tensors="pt", **tokenizer_kwargs
            )
        else:
            prefix = args.prefix
            encoded_prompt = self.tokenizer.encode(prefix + prompt_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(args.device)

        if encoded_prompt.size()[-1] == 0:
            input_ids = None
        else:
            input_ids = encoded_prompt

        output_sequences = self.model.generate(
            input_ids=input_ids,
            max_length=args.length + len(encoded_prompt[0]),
            temperature=args.temperature,
            top_k=args.k,
            top_p=args.p,
            repetition_penalty=args.repetition_penalty,
            do_sample=True,
            num_return_sequences=args.num_return_sequences,
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences = []

        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            generated_sequence = generated_sequence.tolist()

            # Decode text
            text = self.tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

            # Remove all text after the stop token
            text = text[: text.find(args.stop_token) if args.stop_token else None]

            # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
            total_sequence = (
                    prompt_text + text[
                                  len(self.tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)):]
            )

            generated_sequences.append(total_sequence)

        return generated_sequences


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

    def init_gltr_models(self, model: str = DETECTOR_MAP['gltr-detector'][0]):
        if model not in DETECTOR_MAP['gltr-detector']:
            raise RuntimeError

        default_port = DETECTOR_MAP['gltr-detector-server'][DETECTOR_MAP['gltr-detector'].index(model)][-5:-1]
        self.default_logger.info(f"Initialize GLTR {model}")
        gltr_gpt_server = subprocess.Popen(
            ["python", "./reliability_assessment/gltr/server.py", "--model", f"{model}", "--port",
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
        tweets_collection = self.db_instance.get_all_tweets(self.input_date, self.ticker, database='tweet',
                                                            ra_raw=False, feature_filter=False)
        authors_collection = self.db_instance.get_all_authors(self.input_date, self.ticker, database='author')

        # Initialize Feature Records
        self.db_instance.update_all('ra_raw.feature-filter', False, self.input_date, self.ticker)

        # Append a field in the ra_raw.feature-filter
        authors_collection[:] = [author for author in authors_collection if self.__author_feature_rules(author)]
        authors_id = [author['id'] for author in authors_collection]
        tweets_collection[:] = [tweet for tweet in tweets_collection if
                                self.__tweet_feature_rules(tweet) and tweet['author_id'] in authors_id]

        self.db_instance.update_one_bulk([tweet['_id'] for tweet in tweets_collection], 'ra_raw.feature-filter',
                                         [True for _ in range(len(tweets_collection))], self.input_date, self.ticker)

    def detector_wrapper(self, tweet, mode):
        tweet_text = self.__tweet_preprocess(tweet['text'])
        return {'_id': tweet['_id'], 'output': self.nv_instance.detect(text=tweet_text, mode=mode)}

    def neural_fake_news_update(self, gpt_2: bool = False, gltr_gpt2: bool = False, gltr_bert: bool = False, fake: bool = False):
        # Always clean up fields before starting!
        # if input('CAUTION: DO YOU WANT TO CLEAN RA RESULTS? (Y/N) ') == "Y" and input('DOUBLE CHECK (Y/N) ') == 'Y':
        #     self.db_instance.remove_many('ra_raw', self.input_date, self.ticker)

        if gpt_2:
            self.nv_instance.init_gpt_model(model=DETECTOR_MAP['gpt-detector'])
            # Split large tweets collection into smaller pieces -> GOOD FOR LAPTOP :)
            SLICES = 10  # Good for 1080 Ti
            if fake:
                gpt_collection = \
                    self.db_instance.get_neural_non_updated_tweets('ra_raw.RoBERTa-detector',
                                                                   self.input_date, self.ticker,
                                                                   database='fake',
                                                                   select_field={"_id": 1, "id": 1, "text": 1},
                                                                   feature_filter=False)
            else:
                gpt_collection = self.db_instance.get_neural_non_updated_tweets('ra_raw.RoBERTa-detector',
                                                                                self.input_date, self.ticker)
            self.default_logger.info(f'Remaining entries to verify with GPT-2: {len(gpt_collection)}')

            for i in trange(0, len(gpt_collection), SLICES):
                tweets_collection_small = gpt_collection[i:i + SLICES]
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
            SLICES = 4 if gltr_gpt2 else 50
            if fake:
                gltr_collection = self.db_instance.get_neural_non_updated_tweets(
                    f"ra_raw.{gltr_type}-detector",
                    self.input_date, self.ticker,
                    database='fake',
                    select_field={"_id": 1, "id": 1, "text": 1},
                    feature_filter=False)
            else:
                gltr_collection = self.db_instance.get_neural_non_updated_tweets(
                    f"ra_raw.{gltr_type}-detector", self.input_date, self.ticker)
            self.default_logger.info(f'Remaining entries to verify with GLTR: {len(gltr_collection)}')

            for i in trange(0, len(gltr_collection), SLICES):
                tweets_collection_small = gltr_collection[i:i + SLICES]
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

        self.default_logger.info("Neural Fake News Detector Output Update Success!")

    def neural_fake_news_dataset_handle(self):
        tweets_collection = [self.__tweet_preprocess(tweet['text']).replace("\n", "") for tweet in
                             self.db_instance.get_roberta_threshold_tweets(
                                 self.config.getfloat('RA.Neural.Config', 'roberta_threshold'),
                                 self.input_date, self.ticker)]

        train, test = np.split(np.array(tweets_collection), [int(len(tweets_collection) * 0.8)])
        for index, value in {'train': train, 'test': test}.items():
            with open(f'./reliability_assessment/detector_dataset/{self.ticker}_{self.input_date}_{index}.txt', 'w+',
                      encoding='utf-8') as file_handle:
                file_handle.writelines(f"{tweet}\n" for tweet in value)

    def neural_fake_news_generator_fine_tune(self, model_type, model_name_or_path):
        gpt_2_fine_tune = subprocess.call(
            ["python", "./reliability_assessment/run_clm.py", "--model_type", "gpt2",
             "--model_name_or_path", "gpt2-medium",
             "--train_data_file",
             f"./reliability_assessment/detector_dataset/{self.ticker}_{self.input_date}_train.txt",
             "--eval_data_file", f"./reliability_assessment/detector_dataset/{self.ticker}_{self.input_date}_test.txt",
             "--line_by_line", "--do_train", "--do_eval", "--output_dir", "./gpt_generator", "--overwrite_output_dir",
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
        self.db_instance.drop_collection(self.input_date, self.ticker, database='fake')

        tweets_collection = [tweet for tweet in self.db_instance.get_roberta_threshold_tweets(
            self.config.getfloat('RA.Neural.Config', 'roberta_threshold'),
            self.input_date, self.ticker)]

        self.tg_instance.set_model(model_type, model_name_or_path)

        SLICES = 60
        for i in trange(0, len(tweets_collection), SLICES):
            tweets_collection_small = [tweet for tweet in tweets_collection[i:i + SLICES] if
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

    def neural_fake_news_train_classifier(self, gltr_gpt2: bool = False, gltr_bert: bool = False):
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
                                                                     gltr={f"ra_raw.{gltr_type}-detector.frac_hist": 1},
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
        df.to_csv(f'./reliability_assessment/detector_dataset/{self.ticker}_{self.input_date}_{gltr_type}.csv')

        le = preprocessing.LabelEncoder()
        le.fit(y)
        y = le.transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                             'C':      [1, 10, 100, 1000]},
                            {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

        clf = GridSearchCV(SVC(), tuned_parameters, scoring='accuracy', n_jobs=-1, verbose=3, cv=10)
        clf.fit(X_train, y_train)

        self.default_logger.info(clf.best_params_)
        joblib.dump(clf, './reliability_assessment/neural_classifier/{self.ticker}_{self.input_date}_{gltr_type}.pkl')

        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))

    def neural_fake_news_verify(self):
        print("Let's Verify")
