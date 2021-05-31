import pymongo
import configparser

from util import logger


class MongoDB:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('./config.ini')
        client = pymongo.MongoClient(
            f"mongodb+srv://{config.get('MongoDB.Config', 'username')}:{config.get('MongoDB.Config', 'password')}@cluster0.gdbrk.mongodb.net/deeptrust?retryWrites=true&w=majority")
        self.db = client.deeptrust
        self.default_logger = logger.get_logger('mongodb')

    def insert_many(self, record_list, database: str = 'tweet'):
        if database == 'tweet':
            result = self.db.tweet.insert_many(record_list)
        elif database == 'author':
            result = self.db.author.insert_many(record_list)

        self.default_logger.info(f'Insert to {database} with ids {result.inserted_ids}')
