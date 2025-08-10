#!/usr/bin/env python3
"""Prints user locations"""


import sys
import requests
import time


if __name__ == '__main__':
    url = sys.argv[1]
    response = requests.get(url)

    if response.status_code == 403:
        reset = int(response.headers.get('X-Ratelimit-Reset', 0))
        now = int(time.time())
        X = (reset - now) / 60
        print(f"Reset in {X} min")

    if response.status_code == 200:
        data = response.json()
        print(data.get('location'))
    else:
        print("Not found")
        sys.exit(1)
