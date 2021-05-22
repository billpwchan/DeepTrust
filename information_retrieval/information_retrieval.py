from datetime import date

import eikon as ek
import pandas as pd
from searchtweets import ResultStream, collect_results, gen_rule_payload, load_credentials
from OpenPermID import OpenPermID
import json


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

    def get_news(self, start_date: date, end_date: date):
        news_df = ek.get_news_headlines(query='IBM', date_from=start_date, date_to=end_date)
        print(news_df)

    def get_intelligent_tagging(self, query: str):
        output, err = self.opid.calais(query, language='English', contentType='raw', outputFormat='json')
        parsed = json.loads(output)
        print(json.dumps(parsed, indent=4, sort_keys=True))

    def get_open_premid_usage(self) -> pd.DataFrame:
        return self.opid.get_usage()


class InformationRetrieval:
    def __init__(self):
        print("IR Instance Created")

    def initialize_query(self):
        """
        We should have two versions of query, one for Eikon and one for Twitter with verified accounts
        """
