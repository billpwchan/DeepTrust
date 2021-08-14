import re
from datetime import date

import emoji
import pymongo
import configparser
from pymongo import UpdateOne
from pymongo.errors import BulkWriteError

from util import logger


class MongoDB:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('./config.ini')
        client = pymongo.MongoClient(
            f"mongodb+srv://{config.get('MongoDB.Config', 'username')}:{config.get('MongoDB.Config', 'password')}@cluster0.gdbrk.mongodb.net/{config.get('MongoDB.config', 'database')}?retryWrites=true&w=majority")
        self.db = client[config.get('MongoDB.Config', 'database')]
        self.default_logger = logger.get_logger('mongodb')

    def create_collections(self, input_date: date, ticker: str):
        collist = self.db.list_collection_names()
        collection_prefix = f'{ticker}_{input_date.strftime("%Y-%m-%d")}'
        for database in ['tweet', 'author']:
            if f'{collection_prefix}_{database}' in collist:
                self.default_logger.warning(f'{collection_prefix}_{database} collection already exists.')
                self.drop_collection(input_date, ticker, database=database)

        # Ensure Unique Index
        self.db[f'{collection_prefix}_tweet'].create_index("id", unique=True)
        self.db[f'{collection_prefix}_author'].create_index("id", unique=True)

    def duplicate_collection(self, input_date: date, ticker: str, source: str = 'tweet', target: str = 'tweet_dump'):
        collist = self.db.list_collection_names()
        collection_prefix = f'{ticker}_{input_date.strftime("%Y-%m-%d")}'
        if f'{collection_prefix}_{target}' in collist:
            self.default_logger.warning(f'{collection_prefix}_{target} collection already exists.')
            self.drop_collection(input_date, ticker, database=target)
        collist = self.db.list_collection_names()
        if f'{collection_prefix}_{target}' not in collist:
            # Duplicate the tweet database to a clean one for referencing later.
            self.db[f'{collection_prefix}_{source}'].aggregate(
                [{'$match': {}}, {'$out': f'{collection_prefix}_{target}'}])
            self.default_logger.info(f"Cloned collection {collection_prefix}_{source} to {collection_prefix}_{target}")
        else:
            self.default_logger.info(f'Operation Cancelled: {collection_prefix}_{target} collection already exists')

    def drop_collection(self, input_date: date, ticker: str, database: str = 'tweet'):
        collection_name = f'{ticker}_{input_date.strftime("%Y-%m-%d")}_{database}'
        if input(f'CAUTION: DO YOU WANT TO CLEAN {collection_name} Database? (Y/N) ') == "Y" and input(
                'DOUBLE CHECK (Y/N) ') == 'Y':
            self.db[collection_name].drop()

    def get_all_tweets(self, input_date: date, ticker: str, database: str = 'tweet', ra_raw: bool = False,
                       feature_filter: bool = True, neural_filter: bool = False,
                       projection_override: dict = None) -> list:
        collection_name = f'{ticker}_{input_date.strftime("%Y-%m-%d")}_{database}'
        self.default_logger.info(f'Retrieve records from database {collection_name}')
        query_field = {}
        if feature_filter:
            query_field['$and'] = [
                {'ra_raw.feature-filter': {'$exists': True}},
                {'ra_raw.feature-filter': True}]
        if neural_filter:
            query_field['$and'].append({'ra_raw.neural-filter': True})
        unselect_filed = {} if ra_raw else {'ra_raw': 0}
        if projection_override is not None:
            unselect_filed = projection_override
        return [record for record in self.db[collection_name].find(query_field, unselect_filed)]

    def get_non_updated_tweets(self, field, input_date: date, ticker: str, database: str = 'tweet',
                               select_field=None, feature_filter: bool = True):
        if select_field is None:
            select_field = {"_id": 1, "id": 1, "text": 1, "public_metrics": 1}
        collection_name = f'{ticker}_{input_date.strftime("%Y-%m-%d")}_{database}'
        query_field = {"$and": [
            {'ra_raw.feature-filter': {'$exists': True}},
            {'ra_raw.feature-filter': True},
            {field: {'$exists': False}},
        ]} if feature_filter else {"$and": [
            {field: {'$exists': False}}
        ]}
        return [record for record in self.db[collection_name].find(query_field, select_field)]

    def get_roberta_threshold_tweets(self, threshold: float, input_date: date, ticker: str, database: str = 'tweet',
                                     ra_raw: bool = False, gltr: dict = None):
        collection_name = f'{ticker}_{input_date.strftime("%Y-%m-%d")}_{database}'
        self.default_logger.info(f'Retrieve RoBERTa-detector records from database {collection_name}')
        query = {"ra_raw.RoBERTa-detector.real_probability": {"$gte": threshold}}
        unselect_filed = {} if ra_raw else {'ra_raw': 0}
        if gltr is not None:
            unselect_filed = gltr
        return [record for record in self.db[collection_name].find(query, unselect_filed)]

    def get_annotated_tweets(self, query_field: dict, input_date: date, ticker: str, database: str = 'tweet',
                             projection_override: dict = None):
        collection_name = f'{ticker}_{input_date.strftime("%Y-%m-%d")}_{database}'
        self.default_logger.info(f'Retrieve annotation records from database {collection_name}')
        projection_filed = {'text': 1, 'ra_raw.label': 1} if projection_override is None else projection_override
        return [record for record in self.db[collection_name].find(query_field, projection_filed)]

    def get_all_authors(self, input_date: date, ticker: str, database: str = 'author'):
        collection_name = f'{ticker}_{input_date.strftime("%Y-%m-%d")}_{database}'
        self.default_logger.info(f'Retrieve records from database {collection_name}')
        return [record for record in self.db[collection_name].find({})]

    def count_documents(self, input_date: date, ticker: str, database: str = 'tweet') -> int:
        collection_name = f'{ticker}_{input_date.strftime("%Y-%m-%d")}_{database}'
        return self.db[collection_name].find().count()

    def check_record_exists(self, field, value, input_date: date, ticker: str, database: str = 'fake') -> bool:
        collection_name = f'{ticker}_{input_date.strftime("%Y-%m-%d")}_{database}'
        return self.db[collection_name].find({field: value}).count() > 0

    def remove_all(self, field, input_date: date, ticker: str, database: str = 'tweet'):
        collection_name = f'{ticker}_{input_date.strftime("%Y-%m-%d")}_{database}'
        result = self.db[collection_name].update({}, {'$unset': {field: ''}}, multi=True)
        self.default_logger.info(f'Update {result} in {collection_name}')

    def update_all(self, field, entry, input_date: date, ticker: str, database: str = 'tweet'):
        collection_name = f'{ticker}_{input_date.strftime("%Y-%m-%d")}_{database}'
        result = self.db[collection_name].update_many({}, {'$set': {field: entry}})
        self.default_logger.info(f'Update {result.modified_count} records in {collection_name}')

    def update_one(self, ref, field, entry, input_date: date, ticker: str, database: str = 'tweet'):
        collection_name = f'{ticker}_{input_date.strftime("%Y-%m-%d")}_{database}'
        result = self.db[collection_name].update_one({'_id': ref}, {'$set': {field: entry}})
        self.default_logger.info(f'Update {result.modified_count} records in {collection_name}')

    def update_one_bulk(self, ref_list: list, field, entry_list: list, input_date: date, ticker: str,
                        ref_field: str = '_id', database: str = 'tweet'):
        assert len(ref_list) == len(entry_list)
        collection_name = f'{ticker}_{input_date.strftime("%Y-%m-%d")}_{database}'
        operations = [
            UpdateOne({ref_field: ref}, {'$set': {field: entry}}) for ref, entry in
            zip(ref_list, entry_list)
        ]
        result = self.db[collection_name].bulk_write(operations)
        self.default_logger.info(f'Update {result.matched_count} records in {collection_name}')

    def insert_many(self, input_date: date, ticker: str, record_list, database: str = 'tweet'):
        collection_name = f'{ticker}_{input_date.strftime("%Y-%m-%d")}_{database}'
        try:
            result = self.db[collection_name].insert_many(record_list, ordered=False,
                                                          bypass_document_validation=True)
            self.default_logger.info(
                f'Insert to {database} with {len(result.inserted_ids)} ids {result.inserted_ids}')
        except BulkWriteError as e:
            print(e.details)
            self.default_logger.warning("Duplicate Entries detected.")
