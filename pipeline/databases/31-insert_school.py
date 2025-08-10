#!/usr/bin/env python3
""" Inserts new document in collection"""


from pymongo import MongoClient


def insert_school(mongo_collection, **kwargs):
    """Inserts new document in collection"""
    _id = mongo_collection.insert_one(kwargs)
    return _id.inserted_id
