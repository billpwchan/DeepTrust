import pymongo


class mongodb:
    def __init__(self):
        client = pymongo.MongoClient(
            "mongodb+srv://deeptrust:<password>@cluster0.gdbrk.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
        db = client.test
