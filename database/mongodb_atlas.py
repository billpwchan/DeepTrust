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

    def insert_many(self, input_date: date, ticker: str, record_list, database: str = 'tweet'):
        collection_prefix = f'{ticker}_{input_date.strftime("%Y-%m-%d")}'
        if database == 'tweet':
            try:
                result = self.db[f'{collection_prefix}_tweet'].insert_many(record_list)
            except BulkWriteError as e:
                self.default_logger.warn("e.details['writeErrors']")
        elif database == 'author':
            try:
                result = self.db[f'{collection_prefix}_author'].insert_many(record_list)
            except BulkWriteError as e:
                self.default_logger.warn("e.details['writeErrors']")

        self.default_logger.info(f'Insert to {database} with ids {result.inserted_ids}')
