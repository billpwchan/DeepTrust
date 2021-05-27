import configparser
import json
import time
import urllib.parse
from datetime import date, timedelta, datetime
from pathlib import Path

import eikon as ek
import pandas as pd
import requests
from OpenPermID import OpenPermID
from bs4 import BeautifulSoup
# file exists
from tqdm import tqdm, trange


class TwitterAPIInterface:
    def __init__(self, bearer_token):
        """
        Use https://github.com/twitterdev/search-tweets-python to power searches in Twitter API
        """
        self.auth_header = self.__create_headers(bearer_token=bearer_token)

    @staticmethod
    def __create_headers(bearer_token) -> dict:
        return {"Authorization": f"Bearer {bearer_token}"}

    def build_query(self, input_date: date):
        """
        https://developer.twitter.com/en/docs/twitter-api/tweets/search/integrate/build-a-query
        """
        REMOVE_ADS = '-is:nullcast'
        # THESE ARE ALWAYS THE TRUSTED SOURCE!! -> USED FOR QUERY ENHANCEMENT
        VERIFIED_AUTHOR = 'is:verified'
        # Generally, for tweets with URL links, it is more reliable
        HAS_LINKS = 'has:links'
        # For DeepTrust, we must have language constraint to only English
        LANG_EN = 'lang:en'

        search_url = "https://api.twitter.com/2/tweets/search/all"

        query_params = {
            'query': f'stock (twtr OR twtr.k OR twitter) {LANG_EN} {VERIFIED_AUTHOR} {REMOVE_ADS}',
            'end_time': datetime(input_date.year, input_date.month, input_date.day).astimezone().isoformat(),
            'max_results': 10,
            'media.fields': 'url'
        }

        response = requests.request("GET", search_url, headers=self.auth_header, params=query_params)
        print(response.status_code)
        if response.status_code != 200:
            raise Exception(response.status_code, response.text)
        return response.json()


class EikonAPIInterface:
    def __init__(self, ek_api_key, open_permid):
        ek.set_app_key(ek_api_key)
        self.opid = OpenPermID()
        self.opid.set_access_token(open_permid)

    @staticmethod
    def get_ric_symbology(ticker: str) -> str:
        isin = ek.get_symbology(ticker, from_symbol_type='ticker', to_symbol_type='ISIN').loc[ticker, 'ISIN']
        ric = ek.get_symbology(isin, from_symbol_type='ISIN', to_symbol_type='RIC').loc[isin, 'RIC']
        return ric

    @staticmethod
    def get_eikon_news(ric: str, input_date: date, d_days: int = 5) -> list:
        """
        Get Maximum of 100 news (usually covers 3-4 days before the anomaly date), and return a list of documents
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
                        time.sleep(5)
                        soup = BeautifulSoup(news_story, "lxml")
                        output_news.loc[len(output_news.index)] = [soup.get_text(strip=True)]
                    except:
                        continue
                    break
            output_news.to_csv(f'./information_retrieval/news/{ric}_stories_{input_date}.csv', encoding='utf-8')
        else:
            output_news = pd.read_csv(file_path, index_col=0, header=0)
        return output_news['news'].tolist()

    @staticmethod
    def get_company_names(ric: str) -> list:
        company_names_df, err = ek.get_data(instruments=[ric], fields=["TR.CompanyName", "TR.CommonName"])
        return list(set(company_names_df.iloc[0, :].tolist()))

    def get_intelligent_tagging(self, query: str, relevance_threshold: float = 0) -> list:
        """
        Return a list of entities detected using Intelligent Tagging API
        :param query: a document in either HTML or pure text format
        :param relevance_threshold: the minimum relevance needed for an entity to be included in the enhanced keyword list
        :return: a list of dictionaries with {'name', 'relevance', 'type'} attributes
        """
        output, err = None, None
        for retry_limit in range(5):
            try:
                output, err = self.opid.calais(query, language='English', contentType='raw', outputFormat='json')
            except:
                continue
            break
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
        self.input_date = input_date
        self.ric = self.ek_instance.get_ric_symbology(ticker)
        self.company_names = self.ek_instance.get_company_names(self.ric)

    def initialize_query(self, top_n: int = 10):
        """
        We should have two versions of query, one for Eikon and one for Twitter with verified accounts
        """

        # eikon_query_keywords = []
        #
        # # Use Refinitiv News to enhance query keywords
        # # SAMPLE OUTPUT: [{'name': 'Meg Tirrell', 'relevance': 0.2, 'type': 'Person'}]
        # eikon_news = self.ek_instance.get_eikon_news(ric=self.ric, input_date=self.input_date)
        # for news in tqdm(eikon_news):
        #     eikon_query_keywords.extend(self.ek_instance.get_intelligent_tagging(query=news, relevance_threshold=0.5))
        #     time.sleep(5)
        #
        # # Filter duplicate keywords and URLs -> Around 70 keywords for a 5 days reuters search from 227 documents
        # # -> Select top n keywords for expansion
        # eikon_query_keywords = [item for item in Counter(
        #     [item['name'].lower() for item in eikon_query_keywords if item['type'] != 'URL']
        # ).most_common(top_n)]

        # Twitter search for Hashtags in Twitter Verified Accounts
        twitter_hashtags = []

        json_response = self.tw_instance.build_query(self.input_date)
        print(json.dumps(json_response, indent=4, sort_keys=True))
