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

    def insert_many(self, record_list):
        result = self.db.twitter.insert_many(record_list)
        self.default_logger.info(f'Insert to MongoDB Atlas as {result.inserted_ids}')
