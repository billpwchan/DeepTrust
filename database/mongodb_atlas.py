from datetime import date

import pymongo
import configparser

from pymongo.errors import BulkWriteError

from util import logger


class MongoDB:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('./config.ini')
        client = pymongo.MongoClient(
            f"mongodb+srv://{config.get('MongoDB.Config', 'username')}:{config.get('MongoDB.Config', 'password')}@cluster0.gdbrk.mongodb.net/deeptrust?retryWrites=true&w=majority")
        self.db = client[config.get('MongoDB.Config', 'database')]
        self.default_logger = logger.get_logger('mongodb')

    def create_collections(self, input_date: date, ticker: str):
        collist = self.db.list_collection_names()
        collection_prefix = f'{ticker}_{input_date.strftime("%Y-%m-%d")}'
        if f'{collection_prefix}_tweet' in collist:
            self.default_logger.warn(f'{collection_prefix}_tweet collection already exists.')
            if input("Delete? (Y/N) ") == "Y":
                self.db[f'{collection_prefix}_tweet'].drop()

        if f'{collection_prefix}_author' in collist:
            self.default_logger.warn(f'{collection_prefix}_author collection already exists.')
            if input("Delete? (Y/N) ") == "Y":
                self.db[f'{collection_prefix}_author'].drop()

        # Ensure Unique Index
        self.db[f'{collection_prefix}_tweet'].create_index("id", unique=True)
        self.db[f'{collection_prefix}_author'].create_index("id", unique=True)

    def drop_collection(self, input_date: date, ticker: str, database: str = 'tweet'):
        collection_name = f'{ticker}_{input_date.strftime("%Y-%m-%d")}_{database}'
        if input(f'CAUTION: DO YOU WANT TO CLEAN {collection_name} Database? (Y/N) ') == "Y" and input(
                'DOUBLE CHECK (Y/N) ') == 'Y':
            self.db[collection_name].drop()

    def get_all_tweets(self, input_date: date, ticker: str, database: str = 'tweet', ra_raw: bool = False) -> list:
        collection_name = f'{ticker}_{input_date.strftime("%Y-%m-%d")}_{database}'
        self.default_logger.info(f'Retrieve records from database {collection_name}')
        select_filed = {"_id": 1, "text": 1, "public_metrics": 1, "ra_raw": 1} \
            if ra_raw else {"_id": 1, "id": 1, "text": 1, "public_metrics": 1}
        return [record for record in self.db[collection_name].find({}, select_filed)]

    def count_documents(self, input_date: date, ticker: str, database: str = 'tweet') -> int:
        collection_name = f'{ticker}_{input_date.strftime("%Y-%m-%d")}_{database}'
        return self.db[collection_name].find().count()

    def remove_many(self, field, input_date: date, ticker: str, database: str = 'tweet'):
        collection_name = f'{ticker}_{input_date.strftime("%Y-%m-%d")}_{database}'
        self.db[collection_name].update_many({}, {'$unset': {field: ''}})

    def update_one(self, ref, field, entry, input_date: date, ticker: str, database: str = 'tweet'):
        collection_name = f'{ticker}_{input_date.strftime("%Y-%m-%d")}_{database}'
        self.db[collection_name].update_one({'_id': ref}, {'$set': {field: entry}}, upsert=True)
        self.default_logger.info(f'Update {ref} in {collection_name}')

    def insert_many(self, input_date: date, ticker: str, record_list, database: str = 'tweet'):
        collection_name = f'{ticker}_{input_date.strftime("%Y-%m-%d")}_{database}'
        try:
            result = self.db[collection_name].insert_many(record_list, ordered=False,
                                                               bypass_document_validation=True)
            self.default_logger.info(
                f'Insert to {database} with {len(result.inserted_ids)} ids {result.inserted_ids}')
        except BulkWriteError as e:
            self.default_logger.warn("Duplicate Entries detected.")
