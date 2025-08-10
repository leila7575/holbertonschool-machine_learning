#!/usr/bin/env python3
"""provides some stats about Nginx logs stored in MongoDB"""


from pymongo import MongoClient

def main():
    """provides some stats about Nginx logs stored in MongoDB"""
    client = MongoClient("mongodb://localhost:27017/")
    db = client.logs
    collection = db.nginx
    x_logs = collection.count_documents({})
    print(f"{x_logs} logs")

    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    print("Methods:")
    for method in methods:
        method_count = collection.count_documents({"method":method})
        print(f"method {method}: {method_count}")

    status_check = collection.count_documents({"method": "GET", "path": "/status"})
    print(f"{status_check} status check")
