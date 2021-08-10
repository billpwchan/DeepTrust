import configparser
import json
import os
import sys
import time
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from string import punctuation

import eikon as ek
import pandas as pd
import pytz
import requests
from OpenPermID import OpenPermID
from bs4 import BeautifulSoup
from tqdm import tqdm, trange

from database.mongodb_atlas import MongoDB
from util import logger


class TwitterAPIInterface:
    default_logger = logger.get_logger('twitter_api')

    def __init__(self, bearer_token):
        """
        Use https://github.com/twitterdev/search-tweets-python to power searches in Twitter API
        """
        self.auth_header = self.__create_headers(bearer_token=bearer_token)

    @staticmethod
    def __create_headers(bearer_token) -> dict:
        return {"Authorization": f"Bearer {bearer_token}"}

    @staticmethod
    def build_query(input_date: date, market_domain: str, entity_names: list, companies: list, directors: list,
                    ticker: str, enhanced_list: list = None, next_token: str = None, verified: bool = True,
                    max_results: int = 10, d_days: int = 7):
        """
        https://developer.twitter.com/en/docs/twitter-api/tweets/search/integrate/build-a-query
        """

        if enhanced_list is None:
            enhanced_list = []

        start_date = (input_date + timedelta(days=1 - d_days))
        end_date = (input_date + timedelta(days=1))

        # Define query metadata
        market_domain = market_domain
        REMOVE_ADS = '-is:nullcast'
        VERIFIED_AUTHOR = 'is:verified' if verified else ''
        LANG_EN = 'lang:en'
        ORIGINAL_TWEETS = '-is:retweet'
        INDIVIDUAL_TWEET = '-is:reply'

        # Define query keywords
        query_keywords = entity_names + directors + enhanced_list
        query_keywords.extend([f'#{ticker}', f'${ticker}'])
        query_keywords = list(set(query_keywords))

        query_params = {
            'query':        f'({market_domain} OR price) ({" OR ".join(query_keywords)}) {LANG_EN} {REMOVE_ADS} {ORIGINAL_TWEETS} {INDIVIDUAL_TWEET}',
            'expansions':   'author_id',
            'tweet.fields': 'author_id,context_annotations,created_at,entities,id,geo,public_metrics,possibly_sensitive,referenced_tweets,source,text,withheld',
            'user.fields':  'created_at,description,id,location,name,public_metrics,url,username,verified',
            'start_time':   datetime(start_date.year, start_date.month, start_date.day).astimezone(
                pytz.utc).isoformat(),
            'end_time':     datetime(end_date.year, end_date.month, end_date.day).astimezone(pytz.utc).isoformat(),
            'max_results':  max_results,
        }
        TwitterAPIInterface.default_logger.info(f'Twitter Search Query: {query_params["query"]}')
        # Get next page -> Search Pagination in Twitter Dev
        if next_token is not None:
            query_params['next_token'] = next_token

        return query_params

    def tw_search(self, query_params: dict) -> json:
        """
        Return JSON object with fields 'data', 'includes', 'meta' \n
        Data.Fields = ['source', 'id', 'entities', 'text', 'reply_settings', 'context_annotations', 'public_metrics', 'conversation_id', 'referenced_tweets', 'author_id', 'possibly_sensitive', 'created_at', 'lang'] \n
        Meta.Fields = ['newest_id', 'oldest_id', 'next_token', 'result_count']
        Maximum 300 requests in 15 mins -> 503 Response
        :param query_params: Use build_query() function to create a Twitter query
        :return:
        """
        for retry_limit in range(50):
            search_url = "https://api.twitter.com/2/tweets/search/all"
            response = requests.request("GET", search_url, headers=self.auth_header, params=query_params)
            if response.status_code == 200:
                time.sleep(4)
                return response.json()
            else:
                self.default_logger.warn(f'{response.status_code}, {response.text}. Retrying...')
                time.sleep(120)
        self.default_logger.error("Twitter Service Unavailable")

    def tw_lookup(self, ids: str, tweet_fields: str) -> json:
        query_params = {
            "ids":          ids,
            "tweet.fields": tweet_fields
        }
        for retry_limit in range(50):
            search_url = f"https://api.twitter.com/2/tweets"
            response = requests.request("GET", search_url, headers=self.auth_header, params=query_params)
            if response.status_code == 200:
                time.sleep(4)
                return response.json()
            else:
                self.default_logger.warn(f'{response.status_code}, {response.text}. Retrying...')
                time.sleep(120)
        self.default_logger.error("Twitter Service Unavailable")


class EikonAPIInterface:
    def __init__(self, ek_api_key, open_permid):
        ek.set_app_key(ek_api_key)
        self.opid = OpenPermID()
        self.opid.set_access_token(open_permid)

    @staticmethod
    def get_ric_symbology(ticker: str) -> str:
        cusip = ek.get_symbology(ticker, from_symbol_type='ticker', to_symbol_type='CUSIP').loc[ticker, 'CUSIP']
        ric = ek.get_symbology(cusip, from_symbol_type='CUSIP', to_symbol_type='RIC').loc[cusip, 'RIC']
        return ric

    @staticmethod
    def get_eikon_news(ric: str, input_date: date, d_days: int = 7) -> list:
        """
        Get Maximum of 100 news (usually covers 3-4 days before the anomaly date), and return a list of documents
        :param d_days: number of past days news to be included
        :param ric: Reuters RIC code
        :param input_date: The anomaly date
        :return:
        """
        file_path = f'./information_retrieval/news/{ric}_Headlines_{input_date}.csv'
        file_path_check = Path(file_path)
        if not file_path_check.is_file():
            # Collect News from Refinitiv API
            news_headlines_df = pd.concat([ek.get_news_headlines(query=f'R:{ric} AND Language:LEN',
                                                                 date_to=(input_date + timedelta(days=1 - i)).strftime(
                                                                     '%Y-%m-%d'),
                                                                 count=100) for i in trange(d_days)], ignore_index=True)
            news_headlines_df.drop_duplicates(inplace=True)
            news_headlines_df.to_csv(file_path)
        else:
            news_headlines_df = pd.read_csv(file_path, index_col=0, header=0)

        output_news = pd.DataFrame(columns=['news'])
        file_path = f'./information_retrieval/news/{ric}_stories_{input_date}.csv'
        file_path_check = Path(file_path)
        if not file_path_check.is_file():
            for story_id in tqdm(news_headlines_df['storyId'].to_list()):
                for retry_limit in range(5):
                    try:
                        news_story = ek.get_news_story(story_id=story_id)
                        time.sleep(1)
                        soup = BeautifulSoup(news_story, "lxml")
                        output_news.loc[len(output_news.index)] = [soup.get_text(strip=True)]
                        break
                    except:
                        print("Unexpected error:", sys.exc_info()[0])
                        continue
            output_news.to_csv(f'./information_retrieval/news/{ric}_stories_{input_date}.csv', encoding='utf-8')
        else:
            output_news = pd.read_csv(file_path, index_col=0, header=0)
        return output_news['news'].tolist()

    @staticmethod
    def get_company_names(ric: str) -> list:
        company_names_df, err = ek.get_data(instruments=[ric], fields=["TR.CompanyName", "TR.CommonName"])
        return list(set(company_names_df.iloc[0, :].tolist()))

    @staticmethod
    def get_directors_names(ric: str) -> list:
        directors_names_df, err = ek.get_data(instruments=[ric], fields=['TR.OfficerName(RNK=R1:R100)'])
        return directors_names_df['Officer Name'].tolist()

    def get_intelligent_tagging(self, query: str, relevance_threshold: float = 0) -> list:
        """
        Return a list of entities detected using Intelligent Tagging API
        :param query: a document in either HTML or pure text format
        :param relevance_threshold: the minimum relevance needed for an entity to be included in the enhanced keyword list
        :return: a list of dictionaries with {'name', 'relevance', 'type'} attributes
        """
        if len(str(query)) == 0:
            return []
        output, err = None, None
        for retry_limit in range(5):
            try:
                output, err = self.opid.calais(query.encode('utf-8'), language='English', contentType='raw',
                                               outputFormat='json')
                time.sleep(1)
                break
            except:
                print("Unexpected error:", sys.exc_info()[0])
                continue
        enhanced_term_list = []
        if err is None and output is not None:
            news_parsed = json.loads(output)
            enhanced_term_list = []
            for key, value in news_parsed.items():
                # Filter Out Irrelevant Keys (i.e., Doc)
                if '_typeGroup' not in value:
                    continue
                # Named-Entity Recognition
                if value['_typeGroup'] == 'entities' and 'name' in value:
                    # The Resolution in entity should always match with the given ticker
                    enhanced_term_list.extend(
                        [{'name': keywords, 'relevance': value['relevance'], 'type': value['_type']} for keywords in
                         value['name'].split('_') if value['relevance'] >= relevance_threshold])
        return enhanced_term_list

    def get_open_permid_usage(self) -> pd.DataFrame:
        return self.opid.get_usage()


class InformationRetrieval:
    def __init__(self, input_date: date, ticker: str):
        config = configparser.ConfigParser()
        config.read('./config.ini')
        self.ek_instance = EikonAPIInterface(ek_api_key=config.get('Eikon.Config', 'ek_api_key'),
                                             open_permid=config.get('Eikon.Config', 'open_permid'))
        self.tw_instance = TwitterAPIInterface(bearer_token=config.get('Twitter.Config', 'bearer_token'))
        self.db_instance = MongoDB()
        self.db_instance.create_collections(input_date=input_date, ticker=ticker)
        self.input_date = input_date
        self.ticker = ticker
        self.ric = self.ek_instance.get_ric_symbology(self.ticker)
        self.default_logger = logger.get_logger('information_retrieval')

    def get_eikon_keywords(self) -> list:
        eikon_query_entities = []

        # Use Refinitiv News to enhance query keywords
        # SAMPLE OUTPUT: [{'name': 'Meg Tirrell', 'relevance': 0.2, 'type': 'Person'}]
        entity_path = Path.cwd() / 'information_retrieval' / 'news' / f'{self.ric}_entities_{self.input_date}.json'
        if os.path.isfile(entity_path):
            with open(str(entity_path), 'r') as fin:
                eikon_query_entities = json.load(fin)
            return list(eikon_query_entities)

        eikon_news = self.ek_instance.get_eikon_news(ric=self.ric, input_date=self.input_date)
        for news in tqdm(eikon_news):
            eikon_query_entities.extend(self.ek_instance.get_intelligent_tagging(query=news, relevance_threshold=0.5))
            time.sleep(5)

        with open(f'./information_retrieval/news/{self.ric}_entities_{self.input_date}.json', 'w') as fout:
            json.dump(eikon_query_entities, fout)

        return eikon_query_entities

    def retrieve_tweets(self, top_n: int = 10):
        """
        We should have two versions of query, one for Eikon and one for Twitter with verified accounts
        """

        eikon_query_entities = self.get_eikon_keywords()

        # Filter duplicate keywords and URLs -> Select top n keywords for expansion
        eikon_query_keywords = [item[0].strip(punctuation) for item in Counter(
            [item['name'].lower() for item in eikon_query_entities if item['type'] != 'URL']
        ).most_common(top_n)]

        self.default_logger.info(f'Query Expansion Keywords from Reuters News: {eikon_query_keywords}')

        # Get a list of correlated companies & alternative names extracted from Reuters News
        eikon_companies = [f'"{item[0].strip(punctuation)}"' for item in Counter(
            [item['name'].lower() for item in eikon_query_entities if item['type'] == 'Company']
        ).most_common(top_n)]

        self.default_logger.info(f'Query Expansion Companies from Reuters News: {eikon_companies}')

        eikon_directors = [f'"{item.strip(punctuation)}"' for item in self.ek_instance.get_directors_names(self.ric)]
        self.default_logger.info(f'Query Expansion Directors from Reuters News: {eikon_directors}')

        # Common names used on Twitter to refer the entity
        entity_names = [company.lower() for company in self.ek_instance.get_company_names(self.ric)]

        # Twitter search for Hashtags in Twitter Verified Accounts
        twitter_entities = {'cashtags': [], 'annotations': [], 'hashtags': []}
        twitter_enhanced_list = []

        while True:
            tw_query = self.tw_instance.build_query(input_date=self.input_date, market_domain='stock',
                                                    entity_names=entity_names, directors=eikon_directors,
                                                    enhanced_list=twitter_enhanced_list,
                                                    companies=eikon_companies, ticker=self.ticker, verified=True,
                                                    max_results=100)
            tw_response = self.tw_instance.tw_search(tw_query)

            twitter_entities = {'cashtags': [], 'annotations': [], 'hashtags': []}
            # Pseudo-Relevance Feedback Mechanism -> Get Entities identified in Tweets
            for tweet in tw_response['data']:
                if 'entities' in tweet:
                    if 'cashtags' in tweet['entities']:
                        twitter_entities['cashtags'].extend([item['tag'] for item in tweet['entities']['cashtags']])
                    if 'hashtags' in tweet['entities']:
                        twitter_entities['hashtags'].extend([item['tag'] for item in tweet['entities']['hashtags']])
                    if 'annotations' in tweet['entities']:
                        twitter_entities['annotations'].extend(
                            [item['normalized_text'] for item in tweet['entities']['annotations']])

            for key, values in twitter_entities.items():
                twitter_entities[key] = [item[0] for item in
                                         Counter([entry.lower() for entry in values]).most_common(2)]

            original_enhanced_list = twitter_enhanced_list.copy()
            twitter_enhanced_list.extend([f'${cashtag}' for cashtag in twitter_entities['cashtags']] + twitter_entities[
                'annotations'] + [f'#{hashtag}' for hashtag in twitter_entities['hashtags']])
            twitter_enhanced_list = list(set(twitter_enhanced_list))
            if Counter(original_enhanced_list) == Counter(twitter_enhanced_list):
                self.default_logger.info("Twitter PRF Complete")
                break
            self.default_logger.info(f'Query Expansion Keywords from Twitter: {twitter_enhanced_list}')

        self.default_logger.info(f'Finalized Query Expansion Keywords from Twitter: {twitter_enhanced_list}')

        # Start to retrieve tweets using pagination!
        next_token = None
        while True:
            tw_query = self.tw_instance.build_query(input_date=self.input_date, market_domain='stock',
                                                    entity_names=entity_names, directors=eikon_directors,
                                                    enhanced_list=twitter_enhanced_list,
                                                    companies=eikon_companies, ticker=self.ticker, verified=True,
                                                    max_results=100, next_token=next_token)
            tw_response = self.tw_instance.tw_search(tw_query)
            self.db_instance.insert_many(self.input_date, self.ticker, tw_response['data'], 'tweet')
            self.db_instance.insert_many(self.input_date, self.ticker, tw_response['includes']['users'], 'author')
            if 'next_token' in tw_response['meta']:
                next_token = tw_response['meta']['next_token']
            else:
                break
            self.default_logger.info(
                f"Oldest ID: {tw_response['meta']['oldest_id']} Newest ID: {tw_response['meta']['newest_id']}")

    def update_tweets(self):
        tweet_fields = 'possibly_sensitive,geo'
        tweet_ids = [tweet['id'] for tweet in
                     self.db_instance.get_all_tweets(self.input_date, self.ticker, feature_filter=False)]
        self.default_logger.info(f'Remaining Tweet IDs to Update: {len(tweet_ids)}')

        SLICES = 100
        for i in trange(0, len(tweet_ids), SLICES):
            ids_collection_small = tweet_ids[i:i + SLICES]
            ids = ",".join([tweet_id for tweet_id in ids_collection_small])
            tw_response = self.tw_instance.tw_lookup(ids, tweet_fields)

            # Update Possibly Sensitive Field
            self.db_instance.update_one_bulk([tweet['id'] for tweet in tw_response['data']], 'possibly_sensitive',
                                             [tweet['possibly_sensitive'] for tweet in tw_response['data']],
                                             self.input_date, self.ticker, ref_field='id')
            # Update Geo Field
            if any('geo' in tweet for tweet in tw_response['data']):
                self.db_instance.update_one_bulk(
                    [tweet['id'] for tweet in tw_response['data'] if 'geo' in tweet], 'geo',
                    [tweet['geo'] for tweet in tw_response['data'] if 'geo' in tweet],
                    self.input_date, self.ticker, ref_field='id')
