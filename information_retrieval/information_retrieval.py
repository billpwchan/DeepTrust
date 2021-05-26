import configparser
import time
from datetime import date, datetime, timedelta

import eikon as ek
import pandas as pd
from bs4 import BeautifulSoup
from eikon import EikonError
from searchtweets import ResultStream, collect_results, gen_rule_payload, load_credentials
from OpenPermID import OpenPermID
import json
from pathlib import Path
import csv

# file exists
from tqdm import tqdm, trange


class TwitterAPIInterface:
    def __init__(self):
        """
        Use https://github.com/twitterdev/search-tweets-python to power searches in Twitter API
        """
        premium_search_args = load_credentials(filename="./twitter_keys.yaml",
                                               yaml_key="search_tweets_api",
                                               env_overwrite=False)
        rule = gen_rule_payload("beyonce", results_per_call=100)
        tweets = collect_results(rule,
                                 max_results=100,
                                 result_stream_args=premium_search_args)
        for tweet in tweets[0:10]:
            print(tweet.all_text, end='\n\n')

    def build_query(self):
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

        # (apple OR iphone)         ipad


class EikonAPIInterface:
    def __init__(self, ek_api_key, open_premid):
        ek.set_app_key(ek_api_key)
        self.opid = OpenPermID()
        self.opid.set_access_token(open_premid)

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
        return output_news['news'].tolist()

    @staticmethod
    def get_company_names(ric: str) -> list:
        company_names_df, err = ek.get_data(instruments=[ric], fields=["TR.CompanyName", "TR.CommonName"])
        return list(set(company_names_df.iloc[0, :].tolist()))

    def get_intelligent_tagging(self, query: str, relevance_threshold: float = 0) -> list:
        output, err = self.opid.calais(query, language='English', contentType='raw', outputFormat='json')
        news_parsed = json.loads(output)
        enhanced_term_list = []
        for key, value in news_parsed.items():
            # Filter Out Irrelevant Keys (i.e., Doc)
            if '_typeGroup' not in value:
                continue
            # Named-Entity Recognition
            if value['_typeGroup'] == 'entities':
                # The Resolution in entity should always match with the given ticker
                enhanced_term_list.extend(
                    [{'name': keywords, 'relevance': value['relevance'], 'type': value['_type']} for keywords in
                     value['name'].split('_') if value['relevance'] >= relevance_threshold])
        return enhanced_term_list

    def get_open_premid_usage(self) -> pd.DataFrame:
        return self.opid.get_usage()


class InformationRetrieval:
    def __init__(self, input_date: date, ticker: str):
        config = configparser.ConfigParser()
        config.read('./config.ini')
        EK_API_KEY = config.get('Eikon.Config', 'EK_API_KEY')
        OPEN_PREMID = config.get('Eikon.Config', 'OPEN_PREMID')
        self.ek_instance = EikonAPIInterface(ek_api_key=EK_API_KEY, open_premid=OPEN_PREMID)
        # self.tw_instance = TwitterAPIInterface()
        self.input_date = input_date
        self.ric = self.ek_instance.get_ric_symbology(ticker)
        self.company_names = self.ek_instance.get_company_names(self.ric)

    def initialize_query(self):
        """
        We should have two versions of query, one for Eikon and one for Twitter with verified accounts
        """
        query_keywords = []

        # Use Refinitiv News to enhance query keywords
        # SAMPLE OUTPUT: [{'name': 'Meg Tirrell', 'relevance': 0.2, 'type': 'Person'}]
        eikon_news = self.ek_instance.get_eikon_news(ric=self.ric, input_date=self.input_date)
        for news in tqdm(eikon_news):
            query_keywords.extend(self.ek_instance.get_intelligent_tagging(query=news, relevance_threshold=0.2))

        # Twitter ...
        print(query_keywords)

        # tags = self.ek_instance.get_intelligent_tagging(query="TESTING QUERY")
