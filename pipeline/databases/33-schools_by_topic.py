#!/usr/bin/env python3
""" returns the list of school having a specific topic"""


from pymongo import MongoClient
from typing import List, Dict


def schools_by_topic(mongo_collection, topic: str) -> List[Dict]:
    """returns the list of school having a specific topic"""
    query = {"topics": topic}
    return list(mongo_collection.find(query))